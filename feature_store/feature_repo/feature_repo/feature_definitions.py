from datetime import timedelta
from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String


attrition_source = FileSource(
    name="attrition_data_source",
    path="/home/abir-ahmmed/projects/employee-attrition-mlops/feature_store/data/attrition.parquet",
    timestamp_field="event_timestamp",
)


employee = Entity(
    name="employee_id", 
    join_keys=["employee_id"], 
    value_type=ValueType.INT64,
    description="Employee Unique ID"
)


attrition_view = FeatureView(
    name="attrition_stats",
    entities=[employee],
    ttl=timedelta(days=36500),
    schema=[
        Field(name="Age", dtype=Int64),
        Field(name="MonthlyIncome", dtype=Int64),
        Field(name="TotalWorkingYears", dtype=Int64),
        Field(name="YearsAtCompany", dtype=Int64),
        Field(name="YearsInCurrentRole", dtype=Int64),
        Field(name="YearsSinceLastPromotion", dtype=Int64),
        Field(name="MaritalStatus", dtype=String),
        Field(name="Gender", dtype=String),
        Field(name="JobSatisfaction", dtype=Int64),
        Field(name="EnvironmentSatisfaction", dtype=Int64),
        Field(name="RelationshipSatisfaction", dtype=Int64),
        Field(name="WorkLifeBalance", dtype=Int64),
        Field(name="Education", dtype=Int64),
        Field(name="Department", dtype=String),
        Field(name="OverTime", dtype=String),
        Field(name="DistanceFromHome", dtype=Int64),
        
    ],
    online=True,
    source=attrition_source,
    tags={"team": "thesis_final"},
)


attrition_service = FeatureService(
    name="attrition_service",
    features=[attrition_view],
)