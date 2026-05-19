// Khai báo schema mapping để trỏ đúng vào Dataset mà Vertex AI Batch Prediction ghi dữ liệu
const ml_schema = "hospital_ml_outputs_" + dataform.projectConfig.vars.env;

declare({
  database: dataform.projectConfig.defaultDatabase,
  schema: ml_schema,
  name: "ml_forecast_results",
  description:
    "Bảng chứa kết quả dự báo thô được ghi trực tiếp bởi Vertex AI Batch Prediction.",
});
