"""Holds function for defining window in DataFrames."""

from typing import Any, List, Optional, Union

from pyspark import sql
from pyspark.sql import Column, WindowSpec, functions

from butterfree.constants.columns import TIMESTAMP_COLUMN
from butterfree.constants.window_definitions import ALLOWED_WINDOWS


class FrameBoundaries:
    """Utility functions for defining the frame boundaries.

    Args:
        mode: available modes to be used in time aggregations.
        window_definition: time ranges to be used in the windows,
        it can be second(s), minute(s), hour(s), day(s), week(s) and year(s),
    """

    def __init__(self, mode: Optional[str], window_definition: str):
        self.mode = mode
        self.window_definition = window_definition

    @property
    def window_size(self) -> int:
        """Returns window size."""
        if int(self.window_definition.split()[0]) <= 0:
            raise KeyError(f"{self.window_definition} have negative element.")
        return int(self.window_definition.split()[0])

    @property
    def window_unit(self) -> str:
        """Returns window unit."""
        unit = self.window_definition.split()[1]
        if unit not in ALLOWED_WINDOWS and self.mode != "row_windows":
            raise ValueError("Not allowed")

        return unit

    def get(self, window: WindowSpec) -> Any:
        """Returns window with or without the frame boundaries."""
        if self.mode is None:
            return window
        if self.mode == "row_windows":
            span = self.window_size - 1
            return window.rowsBetween(-span, 0)
        if self.mode == "fixed_windows":
            span = ALLOWED_WINDOWS[self.window_unit] * self.window_size
            return window.rangeBetween(-span, 0)


class Window:
    """Utility functions for defining a window specification.

    Args:
        partition_by: he partitioning defined.
        order_by: the ordering defined.
        mode: available modes to be used in time aggregations.
        window_definition: time ranges to be used in the windows, it can be second(s),
            minute(s), hour(s), day(s), week(s) and year(s),
        use_short_name: if True, uses a shorter format for window names (e.g., "7d" 
            instead of "over_7_days_rolling_windows"). Default is False for backward
            compatibility.

    Use the static methods in :class:`Window` to create a :class:`WindowSpec`.
    """

    DEFAULT_SLIDE_DURATION: str = "1 day"

    def __init__(
        self,
        window_definition: str,
        partition_by: Optional[Union[Column, str, List[str]]] = None,
        order_by: Optional[Union[Column, str]] = None,
        mode: Optional[str] = None,
        slide: Optional[str] = None,
        use_short_name: bool = False,
    ):
        self.partition_by = partition_by
        self.order_by = order_by or TIMESTAMP_COLUMN
        self.frame_boundaries = FrameBoundaries(mode, window_definition)
        self.slide = slide or self.DEFAULT_SLIDE_DURATION
        self.use_short_name = use_short_name

    def get_name(self) -> str:
        """Return window suffix name based on passed criteria.
        
        This method can return two different formats for window names:
        
        1. When use_short_name=False (default): Returns the original format like 
           "over_7_days_rolling_windows" for backward compatibility.
        
        2. When use_short_name=True: Returns a shorter format like "7d" to support
           the requested format {name}__{aggregation_name}__{window}{unit}.
        
        Args:
            None
            
        Returns:
            str: The window name in either the original or short format
        """
        if not self.use_short_name:
            # Return the original format for backward compatibility
            return "_".join(
                [
                    "over",
                    f"{self.frame_boundaries.window_size}",
                    f"{self.frame_boundaries.window_unit}",
                    self.frame_boundaries.mode,
                ]
            )
        else:
            # Return the shortened format: {window}{unit}
            # Map time units to their short forms
            unit_map = {
                "second": "s",
                "seconds": "s",
                "minute": "mi",
                "minutes": "mi",
                "hour": "h",
                "hours": "h",
                "day": "d",
                "days": "d",
                "week": "w",
                "weeks": "w",
                "month": "m",
                "months": "m",
                "year": "y",
                "years": "y",
            }
            
            # Get the short form of the unit
            unit = unit_map.get(self.frame_boundaries.window_unit, self.frame_boundaries.window_unit[0])
            
            # Return the shortened format: {window}{unit}
            return f"{self.frame_boundaries.window_size}{unit}"

    def get(self) -> Any:
        """Defines a common window to be used both in time and rows windows."""
        if self.frame_boundaries.mode == "rolling_windows":
            return functions.window(
                TIMESTAMP_COLUMN,
                self.frame_boundaries.window_definition,
                slideDuration=self.slide,
            )
        elif self.order_by == TIMESTAMP_COLUMN:
            w = sql.Window.partitionBy(self.partition_by).orderBy(  # type: ignore
                functions.col(TIMESTAMP_COLUMN).cast("long")
            )
        else:
            w = sql.Window.partitionBy(self.partition_by).orderBy(  # type: ignore
                self.order_by
            )
        return self.frame_boundaries.get(w)

    def build_metadata(self) -> str:
        """Build the metadata for the window."""
        return f"{self.frame_boundaries.window_definition}"
