import genesis as gs

from .base import Base

@ti.data_oriented
class Gravel(Base):
    """
    改良版 Gravel:
    砕石を想定したパラメータ。
    """

    def __init__(
        self,
        E=1e6,
        nu=0.25,
        rho=2000.0,
        friction=0.8,
        restitution=0.15,
        damping=0.05,
        sampler="pbs",
        model="linear",   # or "hertz"
    ):
        super().__init__(E, nu, rho, friction, restitution, damping, sampler, model)
        # 追加でゴツゴツした砕石特有のパラメータを加味してもよい