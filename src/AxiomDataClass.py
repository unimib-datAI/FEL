from dataclasses import dataclass, field
from typing import List, Optional, Protocol, TypeVar

import ltn
from dataclasses_json import DataClassJsonMixin
from ltn.core import Expression


class AxiomImplementationCallable(Protocol):
    def __call__(self, *args: Expression, ) -> Expression: ...


@dataclass
class Variable(DataClassJsonMixin):
    name: str
    training: bool = False
    var: Optional[ltn.Variable] = None


@dataclass
class AxiomInfo(DataClassJsonMixin):
    training: bool
    weight: float
    variables: List[str]
    splits: List[str]
    start: Optional[int] = 0
    stop: Optional[int] = 2**10


@dataclass
class AxiomConfig(DataClassJsonMixin):
    name: str
    infos: AxiomInfo


@dataclass
class Axiom(AxiomConfig):
    ltnAxiom: Optional[AxiomImplementationCallable] = None
    ltnVars: List[Optional[Variable]] = field(default_factory=list)

AxiomBase = TypeVar('AxiomBase', bound=AxiomConfig)
