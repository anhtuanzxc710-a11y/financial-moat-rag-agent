"""Streamlit UI checks with AppTest (no network, no API key)."""
import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _result(name):
    return {"company_name": name, "stock_ticker": name[:4].upper(), "date": f"2026-11-14 {name}", "status": "complete",
            "error": False, "feedback": None, "financial_data": "Current Price: 1", "final_summary": f"Summary {name}",
            "final_sources": [], "market_outlook": {"answer": "M", "sources": []},
            "value_analysis": {"answer": "V", "sources": []}, "devils_advocate": {"answer": "D", "sources": []}}


def test_history_view_switches_analysis_and_marks_current():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["analysis_history"] = [_result("Apple"), _result("Tesla")]
    at.session_state["current_analysis_index"] = 0
    at.run()
    assert at.button(key="view_0").disabled and at.button(key="view_0").label == "✅ Viewing"
    at.button(key="view_1").click().run()
    assert at.session_state["current_analysis_index"] == 1
    assert at.header[0].value == "Analysis for Tesla (TESL)"
    assert at.button(key="view_1").disabled and not at.button(key="view_0").disabled
