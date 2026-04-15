from app_pages.multipage import MultiPage

from app_pages.page_summary import page_summary_body
from app_pages.page_data_analysis import page_data_analysis_body
from app_pages.page_predictor import page_predictor_body
from app_pages.page_model_performance import page_model_performance_body
from app_pages.page_roi_analysis import page_roi_analysis_body
from app_pages.page_hypothesis import page_hypothesis_body

app = MultiPage(app_name="Digital Marketing Conversion Predictor")

app.add_page("Project Summary", page_summary_body)
app.add_page("Customer Behaviour Analysis", page_data_analysis_body)
app.add_page("Conversion Predictor", page_predictor_body)
app.add_page("Model Performance", page_model_performance_body)
app.add_page("Campaign ROI Analysis", page_roi_analysis_body)
app.add_page("Project Hypotheses", page_hypothesis_body)

app.run()
