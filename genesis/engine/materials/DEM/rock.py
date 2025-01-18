import genesis as gs

from .base import Base

@ti.data_oriented
class Rock(Base):
    """
    改良版 Rock:
    岩石を想定したパラメータ。
    """

    def __init__(
        self,
        E=3e9,
        nu=0.25,
        rho=2500.0,
        friction=0.6,
        restitution=0.1,
        damping=0.02,
        sampler="pbs",
        model="linear",  # or "hertz"
    ):
        super().__init__(E, nu, rho, friction, restitution, damping, sampler, model)
        # 必要なら岩石特有の破砕モデルなど実装もありえる
