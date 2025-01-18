import genesis as gs

from .base import Base

@ti.data_oriented
class Soil(Base):
    """
    改良版 Soil:
    土砂を想定したパラメータ。
    model 引数で接触モデルを選択 (default: "linear")
    """

    def __init__(
        self,
        E=5e5,           
        nu=0.3,
        rho=1800.0,
        friction=0.7,
        restitution=0.2,
        damping=0.1,
        sampler="pbs",
        model="linear",  # or "hertz"
    ):
        super().__init__(E, nu, rho, friction, restitution, damping, sampler, model)
        # 追加パラメータ(内部摩擦角、粘着力など)を管理したければここで定義:
        # self._cohesion = 0.0
        # self._phi = 30.0