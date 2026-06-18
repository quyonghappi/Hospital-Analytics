declare({
  database: dataform.projectConfig.defaultDatabase,
  schema: "ml_predictions_dev",
  name: "vertex_batch_predictions_raw",
  description:
    "Raw output từ Vertex AI Batch Prediction. Chứa nested structs của prediction value và array SHAP explanations. Dữ liệu chưa được parse và có duplicate giữa các lần chạy.",
});
