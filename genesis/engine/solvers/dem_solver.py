import numpy as np
import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import DEMEntity
from genesis.engine.states.solvers import DEMSolverState
from .base_solver import Solver


@ti.data_oriented
class DEMSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # ------------------------------------------------------------
        # ユーザ設定パラメータ
        # ------------------------------------------------------------
        self._particle_radius = options.particle_radius
        self._youngs_modulus = options.youngs_modulus   # 弾性係数 (法線方向バネ剛性)
        self._poisson_ratio  = options.poisson_ratio    # 必要ならより厳密な計算に利用
        self._friction_coeff = options.friction_coeff   # 粒子間の摩擦係数
        self._restitution    = options.restitution      # 反発係数
        self._damping        = options.get("damping", 0.0)  # (オプション)速度ダンピング

        self._lower_bound = np.array(options.lower_bound, dtype=float)
        self._upper_bound = np.array(options.upper_bound, dtype=float)

        self._mats = []
        self._mats_idx = []
        self._mats_calc_contact_force = []

        # Spatial Hasher
        self.sh = gu.SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )

        self.setup_boundary()

    def setup_boundary(self):
        self.boundary = CubeBoundary(
            lower=self._lower_bound,
            upper=self._upper_bound,
        )

    def add_material(self, material):
        exist = False
        for mat in self._mats:
            if material == mat:
                material._idx = mat._idx
                exist = True
                break
        if not exist:
            new_id = len(self._mats)
            material._idx = new_id
            self._mats.append(material)
            self._mats_idx.append(new_id)
            self._mats_calc_contact_force.append(material.calc_contact_force)

    def add_entity(self, idx, material, shape, surface):
        self.add_material(material)
        entity = DEMEntity(
            scene=self.scene,
            solver=self, 
            material=material,
            morph=shape,
            surface=surface,
            idx=idx,
            particle_start=self.n_particles
        )
        self._entities.append(entity)
        return entity
    
    def init_particle_fields(self):
        struct_particle_state = ti.types.struct(
            pos=gs.ti_vec3,     # 位置
            vel=gs.ti_vec3,     # 速度
            acc=gs.ti_vec3,     # 加速度(オプション)
            force=gs.ti_vec3,   # 合力(接触力+重力など)
            mass=gs.ti_float,
            radius=gs.ti_float,
        )
        struct_particle_state_ng = ti.types.struct(
            reordered_idx=gs.ti_int,
            active=gs.ti_int,
        )
        struct_particle_info = ti.types.struct(
            mat_idx=gs.ti_int
        )

        self.particles = struct_particle_state.field(
            shape=(self._n_particles,),
            needs_grad=False,
            layout=ti.Layout.SOA
        )
        self.particles_ng = struct_particle_state_ng.field(
            shape=(self._n_particles,),
            needs_grad=False,
            layout=ti.Layout.SOA
        )

        self.particles_info = struct_particle_info.field(
            shape=(self._n_particles,), 
            needs_grad=False,
            layout=ti.Layout.SOA
        )

        self.particles_reordered = struct_particle_state.field(
            shape=(self._n_particles,),
            needs_grad=False,
            layout=ti.Layout.SOA
        )
        self.particles_ng_reordered = struct_particle_state_ng.field(
            shape=(self._n_particles,),
            needs_grad=False,
            layout=ti.Layout.SOA
        )

        # レンダリング用
        struct_particle_state_render = ti.types.struct(
            pos=gs.ti_vec3,
            vel=gs.ti_vec3,
            active=gs.ti_int,
        )
        self.particles_render = struct_particle_state_render.field(
            shape=self._n_particles,
            needs_grad=False,
            layout=ti.Layout.SOA
        )

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        """
        勾配を使わない場合は空のままでOK
        """
        pass

    # ------------------------------------------------------------------------------------
    #                                  build
    # ------------------------------------------------------------------------------------
    def build(self):
        self._n_particles = self.n_particles
        if self.is_active():
            self.sh.build()
            self.init_particle_fields()
            self.init_ckpt()

            # エンティティから粒子情報をソルバに書き込み
            for entity in self._entities:
                entity._add_to_solver()

    # ------------------------------------------------------------------------------------
    #                            エンティティ管理
    # ------------------------------------------------------------------------------------
    def add_entity(self, idx, material, shape, surface):
        entity = DEMEntity(
            scene=self.scene,
            solver=self,
            material=material,
            morph=shape,
            surface=surface,
            idx=idx,
            particle_start=self.n_particles,
        )
        self._entities.append(entity)
        return entity

    def is_active(self):
        return self.n_particles > 0

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        else:
            return sum([entity.n_particles for entity in self._entities])

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound

    # ------------------------------------------------------------------------------------
    #                   粒子ソート & リオーダリング (Spatial Hash)
    # ------------------------------------------------------------------------------------
    @ti.kernel
    def _kernel_reorder_particles(self):
        ti.loop_config(block_dim=256)  # GPUでのblock_sizeを指定 (例: 256)
        for i in range(self._n_particles):
            if self.particles_ng[i].active == 1:
                self.sh.compute_idx_for_particle(i, self.particles[i].pos)
            else:
                # 非アクティブ粒子は無視
                pass

        # 各粒子に対して、ソート後のインデックスを sh.gen_reorder() 的な関数で確定させる
        self.sh.generate_reordered_idx(
            self._n_particles,
            self.particles_ng.active,
            self.particles_ng.reordered_idx
        )

        # 一旦すべてを inactive に初期化
        self.particles_ng_reordered.active.fill(0)

        # 実際にソート後配列へコピー
        for i in range(self._n_particles):
            if self.particles_ng[i].active == 1:
                rid = self.particles_ng[i].reordered_idx
                self.particles_reordered[rid] = self.particles[i]
                self.particles_ng_reordered[rid].active = 1

    @ti.kernel
    def _kernel_copy_from_reordered(self):
        ti.loop_config(block_dim=256)
        for i in range(self._n_particles):
            if self.particles_ng[i].active == 1:
                rid = self.particles_ng[i].reordered_idx
                self.particles[i] = self.particles_reordered[rid]

    # ------------------------------------------------------------------------------------
    #                           粒子間接触力の計算
    # ------------------------------------------------------------------------------------
    @ti.func
    def _task_compute_contact_forces(self, i, j, force_acc: ti.template()):
        pos_i = self.particles_reordered[i].pos
        pos_j = self.particles_reordered[j].pos
        r_i   = self.particles_reordered[i].radius
        r_j   = self.particles_reordered[j].radius

        mat_i = self.particles_info[i].mat_idx
        # mat_j = self.particles_info[j].mat_idx (必要であれば)

        r_ij = pos_j - pos_i
        dist = r_ij.norm()
        overlap = (r_i + r_j) - dist

        if overlap > 0.0 and dist > 1e-12:
            normal = r_ij / dist
            rel_vel = self.particles_reordered[j].vel - self.particles_reordered[i].vel

            # 材質テーブルを参照
            contact_force = ti.Vector.zero(gs.ti_float, 3)
            for k in ti.static(range(len(self._mats_idx))):
                if mat_i == self._mats_idx[k]:
                    contact_force = self._mats_calc_contact_force[k](overlap, normal, rel_vel)
                    break

            force_acc += contact_force

    @ti.kernel
    def _kernel_compute_contact_forces(self):
        """
        近傍探索(SpatialHash)を使い、粒子間の合力を計算。
        """
        ti.loop_config(block_dim=256)

        # 重力を適用
        g = ti.Vector([0.0, -9.8, 0.0])
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active == 1:
                mass_i = self.particles_reordered[i].mass
                self.particles_reordered[i].force = g * mass_i

        # 粒子間衝突力を空間ハッシュで計算
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active == 1:
                force_acc = self.particles_reordered[i].force
                # 近傍半径 (2 * 粒子半径 より少し大きめ)
                search_radius = self._particle_radius * 4.0

                # for_all_neighbors 内部でパーティクル間ループを GPU で並列化
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    search_radius,
                    force_acc,
                    self._task_compute_contact_forces
                )

                # 書き戻し
                self.particles_reordered[i].force = force_acc

    # ------------------------------------------------------------------------------------
    #                             時間積分 (速度, 位置)
    # ------------------------------------------------------------------------------------
    @ti.kernel
    def _kernel_advect_velocity(self):
        ti.loop_config(block_dim=256)
        dt = self._substep_dt
        damping = ti.static(self._damping)  # ダンピング係数を static 化

        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active == 1:
                mass_i  = self.particles_reordered[i].mass
                force_i = self.particles_reordered[i].force

                acc_i = force_i / mass_i
                self.particles_reordered[i].acc = acc_i

                vel_i = self.particles_reordered[i].vel
                if damping > 0.0:
                    vel_i *= ti.exp(-damping * dt)

                vel_i += acc_i * dt
                self.particles_reordered[i].vel = vel_i

    @ti.kernel
    def _kernel_advect_position(self):
        ti.loop_config(block_dim=256)
        dt = self._substep_dt
        e  = ti.static(self._restitution)

        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active == 1:
                pos = self.particles_reordered[i].pos
                vel = self.particles_reordered[i].vel
                r   = self.particles_reordered[i].radius

                new_pos = pos + vel * dt

                # CubeBoundary で衝突処理
                new_pos, new_vel = self.boundary.impose_pos_vel(
                    new_pos, vel,
                    radius=r,
                    restitution=e
                )

                self.particles_reordered[i].pos = new_pos
                self.particles_reordered[i].vel = new_vel

    # ------------------------------------------------------------------------------------
    #                               substep
    # ------------------------------------------------------------------------------------
    def substep_pre_coupling(self, f):
        if self.is_active():
            # (1) 粒子をセルソート (リオーダリング)
            self._kernel_reorder_particles()
            # (2) 粒子間衝突力の計算
            self._kernel_compute_contact_forces()
            # (3) 速度更新
            self._kernel_advect_velocity()

    def substep_post_coupling(self, f):
        if self.is_active():
            # (4) 位置更新 & 境界衝突
            self._kernel_advect_position()
            # (5) ソート前にコピーし直す
            self._kernel_copy_from_reordered()

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling_grad(self, f):
        pass

    # ------------------------------------------------------------------------------------
    #                            レンダリング用フィールド更新
    # ------------------------------------------------------------------------------------
    @ti.kernel
    def _kernel_update_render_fields(self):
        ti.loop_config(block_dim=256)
        for i in range(self._n_particles):
            if self.particles_ng[i].active == 1:
                self.particles_render[i].pos = self.particles[i].pos
                self.particles_render[i].vel = self.particles[i].vel
            else:
                self.particles_render[i].pos = gu.ti_nowhere()
            self.particles_render[i].active = self.particles_ng[i].active

    def update_render_fields(self):
        if self.is_active():
            self._kernel_update_render_fields()

    # ------------------------------------------------------------------------------------
    #                            粒子追加 (例)
    # ------------------------------------------------------------------------------------
    @ti.kernel
    def _kernel_add_particles(
        self,
        active: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        mass: ti.f32,
        radius: ti.f32,
        pos: ti.types.ndarray(),
        mat_idx: ti.i32
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            for k in ti.static(range(3)):
                self.particles[i_global].pos[k] = pos[i, k]
            self.particles[i_global].vel   = ti.Vector.zero(gs.ti_float, 3)
            self.particles[i_global].acc   = ti.Vector.zero(gs.ti_float, 3)
            self.particles[i_global].force = ti.Vector.zero(gs.ti_float, 3)
            self.particles[i_global].mass  = mass
            self.particles[i_global].radius = radius

            self.particles_ng[i_global].active = active
            self.particles_info[i_global].mat_idx = mat_idx

    # ------------------------------------------------------------------------------------
    #                             State Get/Set
    # ------------------------------------------------------------------------------------
    @ti.kernel
    def get_frame(self, pos: ti.types.ndarray(), vel: ti.types.ndarray(), active: ti.types.ndarray()):
        ti.loop_config(block_dim=256)
        for i in range(self._n_particles):
            for j in ti.static(range(3)):
                pos[i, j] = self.particles[i].pos[j]
                vel[i, j] = self.particles[i].vel[j]
            active[i] = self.particles_ng[i].active

    @ti.kernel
    def set_frame(self, pos: ti.types.ndarray(), vel: ti.types.ndarray(), active: ti.types.ndarray()):
        ti.loop_config(block_dim=256)
        for i in range(self._n_particles):
            for j in ti.static(range(3)):
                self.particles[i].pos[j] = pos[i, j]
                self.particles[i].vel[j] = vel[i, j]
            self.particles_ng[i].active = active[i]

    def set_state(self, f, state):
        if self.is_active():
            self.set_frame(state.pos, state.vel, state.active)

    def get_state(self, f):
        if self.is_active():
            state = DEMSolverState(self.scene)
            self.get_frame(state.pos, state.vel, state.active)
            return state
        else:
            return None

    def save_ckpt(self, ckpt_name):
        if self.is_active():
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = dict()
                self._ckpt[ckpt_name]["pos"] = torch.zeros((self.n_particles, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["vel"] = torch.zeros((self.n_particles, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["active"] = torch.zeros((self.n_particles,), dtype=gs.tc_int)

            self._kernel_get_particles(
                0,  # particle_start
                self.n_particles,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"]
            )

            self._kernel_get_particles_active(
                0,
                self.n_particles,
                self._ckpt[ckpt_name]["active"]
            )

            self.copy_frame(self._sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        self.copy_frame(0, self._sim.substeps_local)
        self.copy_grad(0, self._sim.substeps_local)

        if self._sim.requires_grad:
            self.reset_grad_till_frame(self._sim.substeps_local)
            
            # pos
            self._kernel_set_particles_pos(
                0,  # particle_start
                self.n_particles,
                self._ckpt[ckpt_name]["pos"]
            )

            # vel
            self._kernel_set_particles_vel(
                0,
                self.n_particles,
                self._ckpt[ckpt_name]["vel"]
            )

            # active
            self._kernel_set_particles_active(
                0,
                self.n_particles,
                self._ckpt[ckpt_name]["active"]
            )

            for entity in self.entities:
                entity.load_ckpt(ckpt_name=ckpt_name)


    @ti.kernel
    def _kernel_set_particles_pos(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        pos: ti.types.ndarray()
    ):
        for i in range(n_particles):
            i_global = particle_start + i
            for k in ti.static(range(3)):
                self.particles[i_global].pos[k] = pos[i, k]

    @ti.kernel
    def _kernel_set_particles_vel(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        vel: ti.types.ndarray()
    ):
        for i in range(n_particles):
            i_global = particle_start + i
            for k in ti.static(range(3)):
                self.particles[i_global].vel[k] = vel[i, k]

    @ti.kernel
    def _kernel_set_particles_active(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        active: ti.types.ndarray()
    ):
        for i in range(n_particles):
            i_global = particle_start + i
            self.particles_ng[i_global].active = active[i]

    @ti.kernel
    def _kernel_get_particles(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = particle_start + i
            for k in ti.static(range(3)):
                pos[i, k] = self.particles[i_global].pos[k]
                vel[i, k] = self.particles[i_global].vel[k]

    @ti.kernel
    def _kernel_get_particles_active(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        active: ti.types.ndarray()
    ):
        for i in range(n_particles):
            i_global = particle_start + i
            active[i] = self.particles_ng[i_global].active