# dem_entity.py

import taichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.engine.states.entities import DEMEntityState  # 存在しない場合は要定義
from .particle_entity import ParticleEntity


@ti.data_oriented
class DEMEntity(ParticleEntity):
    """
    DEM (Discrete Element Method) の粒子エンティティ例。
    """

    def __init__(
        self,
        scene,
        solver,
        material,
        morph,
        surface,
        idx,
        particle_start,
    ):
        """
        Parameters
        ----------
        scene : シミュレーション全体の Scene
        solver : DEMSolver
            dem_solver.py で定義されるソルバインスタンス
        material : 何らかの Material クラス
            mass, radius, friction, restitution などを持ち得る
        morph : 粒子座標や形状情報を保持するオブジェクト (morph.particles 等)
        surface : レンダリング関連の形状やメッシュ
        idx : エンティティID (任意)
        particle_start : ソルバ全体で見たときの粒子インデックス開始位置
        """
        # ParticleEntity のコンストラクタ呼び出し
        super().__init__(
            scene=scene,
            solver=solver,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=0.0,   # DEMの場合は固定不要なら0.0などでOK
            idx=idx,
            particle_start=particle_start,
            need_skinning=False  # DEMでは頂点/フェイス等不要な場合False
        )

        # morph.particles -> (n,3)の座標がある想定
        if hasattr(morph, "particles"):
            self._particles   = gs.utils.misc.to_gs_tensor(morph.particles)
            self._n_particles = self._particles.shape[0]
        else:
            # 未定義の場合の例
            self._particles   = None
            self._n_particles = 0

        # Material から質量や半径を取得 (必要に応じて)
        self._mass   = getattr(material, "mass",   1.0)
        self._radius = getattr(material, "radius", solver._particle_radius)

        # active フラグ (1=active, 0=inactive)
        self._active = gs.ACTIVE  # (内部的には1)

        # ユーザ入力バッファを簡易用意
        self._tgt = {"pos": None, "vel": None, "act": None}
        self._tgt_keys = ["pos", "vel", "act"]

    def _add_to_solver(self):
        if self._particles is None or self._n_particles == 0:
            return
        
        mat_idx = self._material.idx

        # dem_solver.py に定義されている _kernel_add_particles(...) カーネルを呼ぶ
        self._solver._kernel_add_particles(
            self._active,
            self._particle_start,
            self._n_particles,
            float(self._mass),
            float(self._radius),
            self._particles,
            mat_idx=mat_idx
        )

    @gs.assert_built
    def set_pos(self, f, pos):
        self._solver._kernel_set_particles_pos(
            self._particle_start,
            self._n_particles,
            pos,
        )

    @gs.assert_built
    def set_vel(self, f, vel):
        self._solver._kernel_set_particles_vel(
            self._particle_start,
            self._n_particles,
            vel,
        )

    @gs.assert_built
    def set_active(self, f, active):
        self._solver._kernel_set_particles_active(
            self._particle_start,
            self._n_particles,
            active,
        )

    @ti.kernel
    def get_frame(self, pos: ti.types.ndarray(), vel: ti.types.ndarray()):
        for i in range(self._n_particles):
            i_global = i + self._particle_start
            for j in ti.static(range(3)):
                pos[i, j] = self._solver.particles[i_global].pos[j]
                vel[i, j] = self._solver.particles[i_global].vel[j]

    @ti.kernel
    def _kernel_get_particles(self, particles: ti.types.ndarray()):
        for i in range(self._n_particles):
            i_global = i + self._particle_start
            for j in ti.static(range(3)):
                particles[i, j] = self._solver.particles[i_global].pos[j]

    @gs.assert_built
    def get_state(self):
        state = DEMEntityState(self, self._scene.cur_step_global)
        # ソルバ全体から一括getするのではなく、エンティティ範囲のみ抽出する例:
        self.get_frame(state.pos, state.vel)
        # state.active も管理したい場合は自作カーネルで取得、あるいは solver.get_frame(...) 後に slice でも可

        self._queried_states.append(state)
        return state

    @ti.kernel
    def _kernel_get_mass(self, mass: ti.types.ndarray()):
        total_mass = 0.0
        for i in range(self._n_particles):
            i_global = i + self._particle_start
            total_mass += self._solver.particles[i_global].mass
        mass[0] = total_mass
