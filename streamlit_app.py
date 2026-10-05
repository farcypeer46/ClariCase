"""ClariCase complaint intake.

Run: streamlit run streamlit_app.py
"""

from datetime import datetime

import streamlit as st

from src.app.storage import (LocalStore, SupabaseStore, build_record,
                             normalise_tracking_id)
from src.improved_model_1.predict import load_model, predict

MIN_WORDS = 5

# How many candidates to show, by the top candidate's confidence:
# above 80% -> 1, 50-80% -> top 2, below 50% -> top 3.
SINGLE_ABOVE = 0.80
PAIR_FROM = 0.50

st.set_page_config(page_title="ClariCase", page_icon=":material/forum:",
                   layout="centered")

st.markdown("""
<style>
  .block-container { padding-top: 2.5rem; max-width: 760px; }
  .cc-brand { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em;
              color: #0F766E; margin-bottom: 0.1rem; }
  .cc-tagline { font-size: 1.05rem; color: #475569; margin-bottom: 1.4rem; }
  .cc-steps { display: flex; gap: 0.6rem; flex-wrap: wrap;
              margin-bottom: 1.6rem; }
  .cc-step { flex: 1 1 0; min-width: 150px; background: #FFFFFF;
             border: 1px solid #E2E8F0; border-radius: 12px;
             padding: 0.75rem 0.9rem; }
  .cc-step-num { display: inline-block; width: 1.5rem; height: 1.5rem;
                 border-radius: 50%; background: #0F766E; color: #FFFFFF;
                 font-size: 0.8rem; font-weight: 700; text-align: center;
                 line-height: 1.5rem; margin-right: 0.4rem; }
  .cc-step-title { font-weight: 600; color: #1E293B; }
  .cc-step-text { font-size: 0.85rem; color: #64748B; margin-top: 0.25rem; }
  .cc-badge { display: inline-block; padding: 0.2rem 0.7rem;
              border-radius: 999px; font-size: 0.8rem; font-weight: 700;
              letter-spacing: 0.02em; }
  .cc-routed { background: #DCFCE7; color: #166534; }
  .cc-review { background: #FEF3C7; color: #92400E; }
  .cc-received { background: #E2E8F0; color: #334155; }
  .cc-label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
              letter-spacing: 0.06em; color: #64748B; margin: 0.9rem 0 0.2rem; }
  .cc-team { font-size: 1.45rem; font-weight: 700; color: #1E293B; }
  .cc-issue { font-size: 1.05rem; font-weight: 600; color: #1E293B; }
  .cc-note { font-size: 0.95rem; color: #334155; margin-top: 0.6rem; }
  .cc-footer { text-align: center; font-size: 0.8rem; color: #94A3B8;
               margin-top: 2.5rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_store():
    try:
        cfg = st.secrets["supabase"]
        return SupabaseStore(cfg["url"], cfg["key"])
    except (KeyError, FileNotFoundError):
        return LocalStore()


store = get_store()


# ---------------------------------------------------------------- helpers
def candidates_to_show(candidates: list[dict]) -> list[dict]:
    top = candidates[0]["confidence"]
    n = 1 if top > SINGLE_ABOVE else 2 if top >= PAIR_FROM else 3
    return candidates[:n]


def issue_name(label: str) -> str:
    # "Debt collection :: Communication tactics" -> "Communication tactics"
    return label.split(" :: ", 1)[-1]


def badge(status: str) -> str:
    css = {"Routed": "cc-routed", "Under review": "cc-review"}.get(
        status, "cc-received")
    return f'<span class="cc-badge {css}">{status}</span>'


def label(text: str) -> None:
    st.markdown(f'<div class="cc-label">{text}</div>', unsafe_allow_html=True)


def show_candidates(items: list[tuple[str, float]], css: str) -> None:
    # One candidate: shown as a heading. Several: confidence bars.
    if len(items) == 1:
        st.markdown(f'<div class="{css}">{items[0][0]}</div>',
                    unsafe_allow_html=True)
        return
    for name, conf in items:
        st.progress(min(max(conf, 0.0), 1.0), text=f"{name} — {conf:.0%}")


# ---------------------------------------------------------------- callbacks
# Both run before the rerun, so the input boxes can be cleared on every press.
def handle_submit() -> None:
    text = st.session_state["complaint_text"].strip()
    st.session_state["complaint_text"] = ""
    if len(text.split()) < MIN_WORDS:
        st.session_state["submit_result"] = {"kind": "too_short"}
        return
    prediction = predict(text, model=get_model())
    record = build_record(text, prediction)
    try:
        store.add(record)
    except Exception:
        st.session_state["submit_result"] = {"kind": "error"}
        return
    st.session_state["submit_result"] = {"kind": "sent", "record": record,
                                         "prediction": prediction}


def handle_track() -> None:
    raw = st.session_state["tracking_input"].strip()
    st.session_state["tracking_input"] = ""
    if not raw:
        st.session_state["track_result"] = None
        return
    tracking_id = normalise_tracking_id(raw)
    try:
        found = store.get(tracking_id)
    except Exception:
        st.session_state["track_result"] = {"kind": "error"}
        return
    st.session_state["track_result"] = (
        {"kind": "not_found", "id": tracking_id} if found is None
        else {"kind": "found", "row": found})


# ---------------------------------------------------------------- header
st.markdown('<div class="cc-brand">ClariCase</div>'
            '<div class="cc-tagline">Tell us about a problem with a financial '
            'product or service, in your own words. We\'ll get it to the team '
            'that handles it.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="cc-steps">
  <div class="cc-step"><span class="cc-step-num">1</span>
    <span class="cc-step-title">Describe it</span>
    <div class="cc-step-text">No forms or categories to pick.</div></div>
  <div class="cc-step"><span class="cc-step-num">2</span>
    <span class="cc-step-title">We route it</span>
    <div class="cc-step-text">Matched to the right team and issue.</div></div>
  <div class="cc-step"><span class="cc-step-num">3</span>
    <span class="cc-step-title">Track it</span>
    <div class="cc-step-text">Check progress with your tracking ID.</div></div>
</div>
""", unsafe_allow_html=True)

submit_tab, track_tab = st.tabs([":material/edit_note: Submit a complaint",
                                 ":material/search: Track my complaint"])

# ---------------------------------------------------------------- submit
with submit_tab:
    with st.form("complaint", border=True):
        st.text_area("What happened?", key="complaint_text", height=200,
                     placeholder="For example: I was charged an overdraft fee "
                                 "even though I had enough money in my "
                                 "account. I'd like the fee refunded.")
        st.form_submit_button("Submit complaint", type="primary",
                              icon=":material/send:", on_click=handle_submit,
                              width="stretch")

    result = st.session_state.get("submit_result")
    if result is None:
        pass
    elif result["kind"] == "too_short":
        st.warning(f"Please describe your complaint in at least {MIN_WORDS} "
                   f"words so we can route it correctly.",
                   icon=":material/info:")
    elif result["kind"] == "error":
        st.error("We couldn't save your complaint. Please try again in a "
                 "moment.", icon=":material/error:")
    else:
        record, prediction = result["record"], result["prediction"]
        teams = candidates_to_show(prediction["top_k"])
        issues = candidates_to_show(prediction["issue_top_k"])
        routed = prediction["route"] == "auto"

        with st.container(border=True):
            st.markdown(badge(record["status"]), unsafe_allow_html=True)
            if routed:
                st.markdown('<div class="cc-note">Your complaint has been sent '
                            'to the team that handles it.</div>',
                            unsafe_allow_html=True)
                label("Sent to")
            else:
                st.markdown('<div class="cc-note">Your complaint has been '
                            'received. A specialist will review it to confirm '
                            'the right team.</div>', unsafe_allow_html=True)
                label("Most likely team" if len(teams) == 1
                      else "Most likely teams")
            show_candidates([(t["team_name"], t["confidence"]) for t in teams],
                            "cc-team")

            label("Issue" if len(issues) == 1 else "Possible issues")
            show_candidates([(issue_name(i["issue_label"]), i["confidence"])
                             for i in issues], "cc-issue")
            if len(teams) > 1:
                st.caption(f"Issues shown are for the {teams[0]['team_name']}.")

            label("Your tracking ID")
            st.code(record["tracking_id"], language=None)
            st.caption("Save this ID to check on your complaint in the "
                       "**Track my complaint** tab.")

# ---------------------------------------------------------------- track
with track_tab:
    with st.form("track", border=True):
        st.text_input("Tracking ID", key="tracking_input",
                      placeholder="CC-XXXXXX")
        st.form_submit_button("Check status", icon=":material/search:",
                              on_click=handle_track, width="stretch")

    tracked = st.session_state.get("track_result")
    if tracked is None:
        pass
    elif tracked["kind"] == "error":
        st.error("We couldn't look up your complaint. Please try again in a "
                 "moment.", icon=":material/error:")
    elif tracked["kind"] == "not_found":
        st.error(f"No complaint found with tracking ID {tracked['id']}.",
                 icon=":material/search_off:")
    else:
        found = tracked["row"]
        submitted_at = datetime.fromisoformat(found["submitted_at"])
        with st.container(border=True):
            st.markdown(f"{badge(found['status'])}&nbsp;&nbsp;"
                        f"<code>{found['tracking_id']}</code>",
                        unsafe_allow_html=True)
            # Complaints saved before route/issue existed have None here.
            label("Most likely team" if found.get("route") == "review"
                  else "Sent to")
            st.markdown(f'<div class="cc-team">{found["team_name"]}</div>',
                        unsafe_allow_html=True)
            if found.get("issue_label"):
                label("Issue")
                st.markdown(f'<div class="cc-issue">'
                            f'{issue_name(found["issue_label"])}</div>',
                            unsafe_allow_html=True)
            label("Submitted")
            st.write(f"{submitted_at:%B %d, %Y at %H:%M} UTC")
            with st.expander("Your complaint"):
                st.write(found["complaint_text"])

# ---------------------------------------------------------------- footer
footer = ("Clear-cut complaints are routed automatically; the rest are "
          "reviewed by a person.")
if not store.persistent:
    footer += " · Local mode: complaints are saved on this machine only."
st.markdown(f'<div class="cc-footer">{footer}</div>', unsafe_allow_html=True)
