// Lấy các biến từ workflow_settings.yaml
const env = dataform.projectConfig.vars.env;
const stagingName = dataform.projectConfig.vars.staging_dataset;

const fullStagingSchema = `${stagingName}_${env}`;

// Khai báo các bảng External từ Terraform
declare({
  database: dataform.projectConfig.defaultProject,
  schema: fullStagingSchema,
  name: "ext_fact_hospital_utilization",
  description: "External table trỏ tới GCS Silver Zone - Fact Data",
});

declare({
  database: dataform.projectConfig.defaultProject,
  schema: fullStagingSchema,
  name: "ext_dim_hospital",
  description: "External table trỏ tới GCS Silver Zone - Dimension Data",
});

declare({
  database: dataform.projectConfig.defaultProject,
  schema: fullStagingSchema,
  name: "ext_dim_date",
  description: "External table trỏ tới GCS Silver Zone - Dimension Data",
});

declare({
  database: dataform.projectConfig.defaultProject,
  schema: fullStagingSchema,
  name: "ext_dim_geography",
  description: "External table trỏ tới GCS Silver Zone - Dimension Data",
});

declare({
  database: dataform.projectConfig.defaultProject,
  schema: fullStagingSchema,
  name: "ext_dim_population",
  description: "External table trỏ tới GCS Silver Zone - Dimension Data",
});
