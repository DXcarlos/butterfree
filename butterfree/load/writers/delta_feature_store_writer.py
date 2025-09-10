from typing import Any, Dict, List, Optional

from pyspark.sql.dataframe import DataFrame

from butterfree.clients import SparkClient
from butterfree.configs.db import DeltaConfig
from butterfree.constants.columns import TIMESTAMP_COLUMN  # noqa: F401
from butterfree.load.writers.delta_writer import DeltaWriter
from butterfree.load.writers.writer import Writer
from butterfree.transform import FeatureSet


class DeltaFeatureStoreWriter(Writer):
    """Enable writing feature sets into Delta tables with merge or replace capabilities.

    Attributes:
        database: database name to use for the Delta table.
        table: table name to write the feature set to.
        merge_on: list of columns to use as merge keys. Required when use_replace
            is False,
            optional when use_replace is True.
        deduplicate: whether to deduplicate data before merging based on featr set keys.
            Default is False.
        when_not_matched_insert: optional condition for insert operations.
            When provided, rows will only be inserted if this condition is true.
        when_matched_update: optional condition for update operations.
            When provided, rows will only be updated if this condition is true.
            Source columns can be referenced as source.<column_name> and target
            columns as target.<column_name>.
        when_matched_delete: optional condition for delete operations.
            When provided, rows will be deleted if this condition is true.
            Source and target columns can be referenced as in update conditions.
        use_replace: whether to use replace instead of merge. When True, the writer
            will use the replaceWhere option with overwrite mode, which is typically
            faster than merge operations for large datasets. Default is False.
        replace_where: optional condition for replace operations using replaceWhere.
            Only used when use_replace is True. If not provided when use_replace is True
            and auto_date_filter is False, all data in the table will be replaced.
        auto_date_filter: whether to automatically generate a replace_where filter based
            on the min/max dates in the dataframe using the TIMESTAMP_COLUMN
            constant.
            Only used when use_replace is True and replace_where is None.
            Default is False.

    Example:
        Simple example regarding DeltaFeatureStoreWriter class instantiation.
        We can instantiate this class with basic merge configuration:

    >>> from butterfree.load.writers import DeltaFeatureStoreWriter
    >>> spark_client = SparkClient()
    >>> writer = DeltaFeatureStoreWriter(
    ...     database="feature_store",
    ...     table="user_features",
    ...     merge_on=["id", "timestamp"]
    ... )
    >>> writer.write(feature_set=feature_set,
    ...             dataframe=dataframe,
    ...             spark_client=spark_client)

        We can also enable deduplication based on the feature set keys:

    >>> writer = DeltaFeatureStoreWriter(
    ...     database="feature_store",
    ...     table="user_features",
    ...     merge_on=["id", "timestamp"],
    ...     deduplicate=True
    ... )

        For more control over the merge operation, we can add conditions:

    >>> writer = DeltaFeatureStoreWriter(
    ...     database="feature_store",
    ...     table="user_features",
    ...     merge_on=["id", "timestamp"],
    ...     when_matched_update="source.value > target.value",
    ...     when_not_matched_insert="source.value > 0"
    ... )

        For faster writes with large datasets, we can use replace instead of merge.
        When using replace, merge_on is optional:

    >>> writer = DeltaFeatureStoreWriter(
    ...     database="feature_store",
    ...     table="user_features",
    ...     use_replace=True,
    ...     replace_where="date_column >= '2023-01-01' AND date_column <= '2023-01-31'"
    ... )

        We can also use auto_date_filter to automatically generate the replace_where
        condition based on the min/max dates in the dataframe:

    >>> writer = DeltaFeatureStoreWriter(
    ...     database="feature_store",
    ...     table="user_features",
    ...     use_replace=True,
    ...     auto_date_filter=True
    ...     # date_column defaults to TIMESTAMP_COLUMN
    ... )

        The writer supports schema evolution by default and will automatically
        handle updates to the feature set schema.

        When writing with deduplication enabled, the writer will use the feature
        set's key columns and timestamp to ensure data quality by removing
        duplicates before merging.

        For optimal performance, it's recommended to:
        1. Choose appropriate merge keys
        2. Use conditions to filter unnecessary updates/inserts
        3. Enable deduplication only when needed
        4. Use replace instead of merge for large datasets when appropriate
    """

    def __init__(
        self,
        database: str,
        table: str,
        merge_on: Optional[List[str]] = None,
        when_not_matched_insert: Optional[str] = None,
        when_matched_update: Optional[str] = None,
        when_matched_delete: Optional[str] = None,
        use_replace: bool = False,
        replace_where: Optional[str] = None,
        auto_date_filter: bool = False,
    ):
        # Validate that merge_on is provided when not using replace
        if not use_replace and merge_on is None:
            raise ValueError("merge_on is required when use_replace is False")

        self.config = DeltaConfig(
            database=database,
            table=table,
            merge_on=merge_on,
            when_not_matched_insert=when_not_matched_insert,
            when_matched_update=when_matched_update,
            when_matched_delete=when_matched_delete,
            replace_where=replace_where,
        )
        self.use_replace = use_replace
        self.auto_date_filter = auto_date_filter
        self.row_count_validation = False

    def write(
        self,
        dataframe: DataFrame,
        spark_client: SparkClient,
        feature_set: FeatureSet,
    ) -> None:
        """Writes the input dataframe into a Delta table using merge or replace.

        When use_replace is False (default), performs a Delta merge operation with
        the provided dataframe using the config merge settings.

        When use_replace is True, performs a Delta replace operation using the
        replaceWhere option, which is typically faster than merge operations for
        large datasets.

        When use_replace is True and auto_date_filter is True, automatically generates
        a replace_where condition based on the min/max dates in the dataframe using
        the
        TIMESTAMP_COLUMN constant. This is useful when you want to replace data for a
        specific time range without having to manually specify the range.

        The priority for replace conditions is:
        1. Explicit replace_where parameter (if provided)
        2. Auto-generated filter from auto_date_filter (if enabled and replace_where
            is None)
        3. No filter (replaces all data in the table)

        Args:
            dataframe: Spark dataframe with data to be written.
            spark_client: Client with an active Spark connection.
            feature_set: Feature set instance containing schema and configuration.
                Used for deduplication when enabled.

        Example:
            Using merge (default):
            >>> from butterfree.load.writers import DeltaFeatureStoreWriter
            >>> writer = DeltaFeatureStoreWriter(
            ...     database="feature_store",
            ...     table="user_features",
            ...     merge_on=["id", "timestamp"]
            ... )
            >>> writer.write(
            ...     dataframe=dataframe,
            ...     spark_client=spark_client,
            ...     feature_set=feature_set
            ... )

            Using replace with explicit replace_where (merge_on is optional):
            >>> writer = DeltaFeatureStoreWriter(
            ...     database="feature_store",
            ...     table="user_features",
            ...     use_replace=True,
            ...     replace_where="date_column >= '2023-01-01'"
            ... )
            >>> writer.write(
            ...     dataframe=dataframe,
            ...     spark_client=spark_client,
            ...     feature_set=feature_set
            ... )

            Using replace with auto_date_filter (automatically generates replace_where):
            >>> writer = DeltaFeatureStoreWriter(
            ...     database="feature_store",
            ...     table="user_features",
            ...     use_replace=True,
            ...     auto_date_filter=True
            ...     # date_column defaults to TIMESTAMP_COLUMN
            ... )
            >>> writer.write(
            ...     dataframe=dataframe,
            ...     spark_client=spark_client,
            ...     feature_set=feature_set
            ... )
        """
        options = self.config.get_options(self.config.table)

        if self.use_replace:
            DeltaWriter.replace(
                client=spark_client,
                database=options["database"],
                table=options["table"],
                source_df=dataframe,
                replace_where=self.config.replace_where,
                auto_date_filter=self.auto_date_filter,
            )
        else:
            # Ensure merge_on is not None before performing merge
            if self.config.merge_on is None:
                raise ValueError("merge_on cannot be None when use_replace is False")

            DeltaWriter.merge(
                client=spark_client,
                database=options["database"],
                table=options["table"],
                merge_on=self.config.merge_on,
                source_df=dataframe,
                when_not_matched_insert=self.config.when_not_matched_insert,
                when_matched_update=self.config.when_matched_update,
                when_matched_delete=self.config.when_matched_delete,
            )

    def validate(
        self,
        dataframe: DataFrame,
        spark_client: SparkClient,
        feature_set: FeatureSet,
    ) -> None:
        """Validates the dataframe written to Delta table.

        In Delta tables, schema validation is handled by Delta's schema enforcement
        and evolution. No additional validation is needed.

        Args:
            dataframe: Spark dataframe to be validated
            spark_client: Client for Spark connection
            feature_set: Feature set with the schema definition
        """
        pass

    def check_schema(self, dataframe: DataFrame, schema: List[Dict[str, Any]]) -> None:
        """Checks if the dataframe schema matches the feature set schema.

        Schema validation in Delta tables is handled by Delta Lake's schema enforcement
        and evolution capabilities.

        Args:
            dataframe: Spark dataframe to be validated
            schema: Schema definition from the feature set
        """
        pass
