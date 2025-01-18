# base.py
import taichi as ti
import genesis as gs
from ..base import Material


@ti.data_oriented
class Base(Material):
    """
    DEM 用のベースマテリアルクラス(改良版)。
    - 複数の接触モデルをサポートし、model 引数で切り替えられるようにする。
    - ヤング率(E), ポアソン比(nu), 密度(rho),
      摩擦係数(friction), 反発係数(restitution), ダンピング(damping)
      などの基本パラメータを保持。

    例: model="linear" -> 線形バネ+クーロン摩擦+ダンピング
        model="hertz"  -> Hertz-Mindlin 近似 (Hertz接触+クーロン摩擦など) (仮)
    """

    def __init__(
        self,
        E=1e6,
        nu=0.2,
        rho=1000.0,
        friction=0.5,
        restitution=0.3,
        damping=0.0,
        sampler="pbs",
        model="linear",
    ):
        super().__init__()

        self._E = E
        self._nu = nu
        self._rho = rho
        self._fric = friction
        self._rest = restitution
        self._damp = damping
        self._sampler = sampler
        self._idx = None

        if model == "linear":
            # 線形(デフォルト)モデルにする
            self.calc_contact_force = self.calc_contact_force_linear
        elif model == "hertz":
            # Hertz 的モデルにする (簡易実装例)
            self.calc_contact_force = self.calc_contact_force_hertz
        else:
            gs.raise_exception(f"Unrecognized DEM contact model: {model}")

        self._model = model

    @classmethod
    def _repr_type(cls):
        return f"<gs.materials.DEM.{cls.__name__}>"

    # -----------------------------------------------
    # 各種接触モデル
    # -----------------------------------------------
    @ti.func
    def calc_contact_force_linear(self, overlap, normal, rel_vel):
        """
        線形バネ + クーロン摩擦 + ダンピング の簡易実装。
        """
        if overlap <= 0.0:
            return ti.Vector.zero(gs.ti_float, 3)

        # 線形バネ剛性 (単純に E を用いる)
        k_n = self._E
        f_n_mag = k_n * overlap
        f_n = f_n_mag * normal

        # クーロン摩擦
        v_n = rel_vel.dot(normal) * normal
        v_t = rel_vel - v_n
        f_t_mag = self._fric * f_n_mag
        if v_t.norm() > 1e-8:
            f_t = -f_t_mag * v_t.normalized()
        else:
            f_t = ti.Vector.zero(gs.ti_float, 3)

        # ダンピング
        f_damp = -self._damp * rel_vel

        return f_n + f_t + f_damp

    @ti.func
    def calc_contact_force_hertz(self, overlap, normal, rel_vel):
        """
        Hertz(半球接触) + Mindlin(せん断)近似 + ダンピング, などをざっくり実装した例(簡易)。
        本格的にはヤング率, ポアソン比から接触半径, 有効半径など計算しきちんとモデル化が必要。
        ここでは非常に簡略化。

        参考: Hertz-Mindlin 接触力 = k_n * overlap^(3/2), など。
        """
        if overlap <= 0.0:
            return ti.Vector.zero(gs.ti_float, 3)

        # Hertzモデルの法線剛性 (簡易例)
        # 真面目には k_n = 4/3 * E_eff * sqrt(R_eff) などの式がある。
        # ここでは overlap^(3/2) を使った簡易近似とし、_Eを係数に用いる。
        k_n = self._E * overlap**0.5  # sqrt(overlap)を剛性に掛ける程度の簡易
        f_n_mag = k_n * overlap**1.5
        f_n = f_n_mag * normal

        # クーロン摩擦(同様)
        v_n = rel_vel.dot(normal) * normal
        v_t = rel_vel - v_n
        f_t_mag = self._fric * f_n_mag
        if v_t.norm() > 1e-8:
            f_t = -f_t_mag * v_t.normalized()
        else:
            f_t = ti.Vector.zero(gs.ti_float, 3)

        # ダンピング
        f_damp = -self._damp * rel_vel

        return f_n + f_t + f_damp

    # -----------------------------------------------
    # eq, property, etc.
    # -----------------------------------------------
    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return (
                (self._E == other._E) and
                (self._nu == other._nu) and
                (self._rho == other._rho) and
                (self._fric == other._fric) and
                (self._rest == other._rest) and
                (self._damp == other._damp) and
                (self._model == other._model)
            )
        return False

    @property
    def idx(self):
        return self._idx

    @property
    def E(self):
        return self._E

    @property
    def nu(self):
        return self._nu

    @property
    def rho(self):
        return self._rho

    @property
    def friction(self):
        return self._fric

    @property
    def restitution(self):
        return self._rest

    @property
    def damping(self):
        return self._damp

    @property
    def sampler(self):
        return self._sampler

    @property
    def model(self):
        """
        DEMの接触モデル名("linear" or "hertz"など)
        """
        return self._model
