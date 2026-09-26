from __future__ import annotations

from enum import Enum


class ActionRepresentation(str, Enum):
    """Physical action representations supported by the WAM-to-SONIC bridge."""

    SONIC_TOKEN = "sonic_token"
    ABSOLUTE = "absolute"
    ABSOLUTE_WITH_RELATIVE_VYAW = "absolute_with_relative_vyaw"
    RELATIVE = "relative"
    RELATIVE_WITH_ABSOLUTE_VYAW = "relative_with_absolute_vyaw"

    @classmethod
    def parse(cls, value: str | None) -> "ActionRepresentation":
        normalized = str(value or cls.SONIC_TOKEN.value).strip().lower()
        aliases = {
            "token": cls.SONIC_TOKEN,
            "abs": cls.ABSOLUTE,
            "abs_rel_vyaw": cls.ABSOLUTE_WITH_RELATIVE_VYAW,
            "absolute_rel_vyaw": cls.ABSOLUTE_WITH_RELATIVE_VYAW,
            "rel": cls.RELATIVE,
            "relative_abs_vyaw": cls.RELATIVE_WITH_ABSOLUTE_VYAW,
            "rel_abs_vyaw": cls.RELATIVE_WITH_ABSOLUTE_VYAW,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"unsupported output_representation={value!r}; expected one of {choices}") from exc

    def relative_joint_ranges(self, *, relative_hands: bool = True) -> tuple[tuple[int, int], ...]:
        """Return WB semantic ranges restored against the chunk's first state.

        The latest WB semantic layout is body q29, root roll/pitch/vyaw3,
        left hand20, right hand20. Yaw angular velocity is semantic index 31.
        """

        if self == self.ABSOLUTE:
            return ()
        if self == self.ABSOLUTE_WITH_RELATIVE_VYAW:
            return ((31, 32),)
        if self == self.RELATIVE_WITH_ABSOLUTE_VYAW:
            return ((0, 31), (32, 72)) if relative_hands else ((0, 31),)
        if self == self.RELATIVE:
            return ((0, 72),) if relative_hands else ((0, 32),)
        raise ValueError("sonic_token has no physical relative_joint_ranges")
