import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
import json


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


SECTOR_PATHS = {
    "Manufacturing": r"Manufacturing_Sector 1.xlsx",
    "Cyber Security": None,
    "Others": None
}
# SHEET_NAME will be dynamic, selected by user


class ComparisonState(TypedDict):
    input_tool: Dict
    compare_tools: List[Dict]
    report: str
    estimated_tokens: int
    actual_tokens: Dict[str, int]
    estimation_done: bool  

def prepare_data(state: ComparisonState) -> ComparisonState:
    if state.get("estimation_done", False):
        # Skip: Already estimated in manual call
        return state
    
    # Estimate input tokens (build prompt_text only once)
    selected_json = json.dumps(state["input_tool"], separators=(',', ':'))
    data_json = json.dumps(state["compare_tools"], separators=(',', ':'))
    prompt_text = f"""As an enterprise software expert, generate a structured comparison report for {selected_json} vs. other PLM/CAD tools from {data_json}. Rate each tool (1-5 scale, 1=poor/5=excellent) on these dimensions, based solely on verifiable data (e.g., vendor sites, Gartner, case studies—no speculation):

- Market Presence (10%): Vendor strength, coverage.
- Cost (20%): Affordability/value.
- Integration (25%): ERP/PLM compatibility (# integrations).
- Features (30%): Capabilities (modeling, collaboration, BOM).
- Efficiency (15%): Time-to-value (deployment, ROI).
- Sources (0%): For validation only.

Research and cover each of these Key Analysis Dimensions:
- SWOT: Strengths, Weaknesses, Opportunities, Threats for the selected tool relative to the top 5 tools.
- GAP Analysis: Feature/innovation gaps between the selected tool and the top 5 tools.
- Market Share: Current shares (e.g., % global/regional), growth rates, and forecasts (2024-2028) for the selected tool and top 5 tools.

Internally calculate weighted scores for all tools [(Score * Weight), sum to total/5]. Rank all by descending total score. STRICTLY DO NOT output any tables, rankings, lists of tools, raw scores, or dimension ratings—keep ALL calculations and details internal. Output ONLY the focused comparison sections below, in plain text with markdown headers.

Focus output EXCLUSIVELY on {selected_json} vs. top 5 ranked tools (exclude selected if not in top 5). Structure output exactly as:

**Market Share Overview**: 3-5 bullets with current % global/regional shares, growth rates, and forecasts (2024-2028) for selected vs. top 5 tools collectively, data-backed (e.g., "Selected holds 5% global share, trailing leaders' 15-25% with 8% CAGR vs. 12% forecast").

**SWOT Analysis**: Present as a clean 2x2 quadrant table in markdown format. Use simple bullet points inside each cell with line breaks (no HTML tags like <br>). Example format:

| **Strengths** | **Weaknesses** |
|---------------|----------------|
| - 2-3 bullets, data-backed (e.g., "Superior BOM features per Gartner").<br>- Bullet 2.<br>- Bullet 3. | - 2-3 bullets, data-backed (e.g., "Limited ERP integrations vs. Siemens").<br>- Bullet 2.<br>- Bullet 3. |
| **Opportunities** | **Threats** |
| - 2-3 bullets, data-backed (e.g., "Growing infrastructure demand aligns with niche strengths").<br>- Bullet 2.<br>- Bullet 3. | - 2-3 bullets, data-backed (e.g., "AI advancements in competitors like Autodesk").<br>- Bullet 2.<br>- Bullet 3. |

**GAP Analysis**: 3-5 consultant-style bullet points on feature/innovation gaps of selected vs. top 5 collectively, data-backed and action-oriented (e.g., "- **AI Modeling Gap**: Selected lacks generative design (Autodesk strength per case studies); recommend R&D investment to close 20% innovation deficit.").

**Recommendation** (short and crisp, 50-100 words): As final conclusion, name best tool from top 5 (or selected if #1). Highlight selected's key strengths/weaknesses vs. best. Justify with 1-2 data points. End with yes/no on switching, tied to enterprise needs (e.g., "Switch to Siemens for scale; retain for niche efficiency")."""
    total_chars = len(prompt_text)
    state["estimated_tokens"] = int(total_chars / 4) + 50  
    state["estimation_done"] = True 
    return state


def generate_comparison(state: ComparisonState) -> ComparisonState:
    chat_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=1500
    )

    prompt = ChatPromptTemplate.from_template("""As an enterprise software expert, generate a structured comparison report for {selected_tool} vs. other PLM/CAD tools from {data_str}. Rate each tool (1-5 scale, 1=poor/5=excellent) on these dimensions, based solely on verifiable data (e.g., vendor sites, Gartner, case studies—no speculation):

- Market Presence (10%): Vendor strength, coverage.
- Cost (20%): Affordability/value.
- Integration (25%): ERP/PLM compatibility (# integrations).
- Features (30%): Capabilities (modeling, collaboration, BOM).
- Efficiency (15%): Time-to-value (deployment, ROI).
- Sources (0%): For validation only.

Research and cover each of these Key Analysis Dimensions:
- SWOT: Strengths, Weaknesses, Opportunities, Threats for the selected tool relative to the top 5 tools.
- GAP Analysis: Feature/innovation gaps between the selected tool and the top 5 tools.
- Market Share: Current shares (e.g., % global/regional), growth rates, and forecasts (2024-2028) for the selected tool and top 5 tools.

Internally calculate weighted scores for all tools [(Score * Weight), sum to total/5]. Rank all by descending total score. STRICTLY DO NOT output any tables, rankings, lists of tools, raw scores, or dimension ratings—keep ALL calculations and details internal. Output ONLY the focused comparison sections below, in plain text with markdown headers.

Focus output EXCLUSIVELY on {selected_tool} vs. top 5 ranked tools (exclude selected if not in top 5). Structure output exactly as:

**Market Share Overview**: 3-5 bullets with current % global/regional shares, growth rates, and forecasts (2024-2028) for selected vs. top 5 tools collectively, data-backed (e.g., "Selected holds 5% global share, trailing leaders' 15-25% with 8% CAGR vs. 12% forecast").

**SWOT Analysis**: Present as a clean 2x2 quadrant table in markdown format. Use simple bullet points inside each cell with line breaks for multi-line content (avoid any HTML tags like <br>; rely on markdown line breaks). Ensure bullets are clean and readable in markdown. Example format exactly:

| **Strengths** | **Weaknesses** |
|---------------|----------------|
| - 2-3 bullets, data-backed (e.g., "Superior BOM features per Gartner").<br>- Bullet 2.<br>- Bullet 3. | - 2-3 bullets, data-backed (e.g., "Limited ERP integrations vs. Siemens").<br>- Bullet 2.<br>- Bullet 3. |
| **Opportunities** | **Threats** |
| - 2-3 bullets, data-backed (e.g., "Growing infrastructure demand aligns with niche strengths").<br>- Bullet 2.<br>- Bullet 3. | - 2-3 bullets, data-backed (e.g., "AI advancements in competitors like Autodesk").<br>- Bullet 2.<br>- Bullet 3. |

**GAP Analysis**: 3-5 consultant-style bullet points on feature/innovation gaps of selected vs. top 5 collectively, data-backed and action-oriented (e.g., "- **AI Modeling Gap**: Selected lacks generative design (Autodesk strength per case studies); recommend R&D investment to close 20% innovation deficit.").

**Recommendation**: 3-5 actionable bullet points as the final conclusion. Name the best tool from top 5 (or selected if #1) in the first bullet. Highlight selected's key strengths/weaknesses vs. best with data-backed justification in subsequent bullets. End with a yes/no on switching, tied to enterprise needs (e.g., "- **Switch Recommendation**: Yes, to Siemens for scale; retain for niche efficiency per ROI case studies."). Keep each bullet concise and action-oriented.""")

    chain = (
        {
            "selected_tool": lambda x: json.dumps(x["input_tool"], separators=(',', ':')),
            "data_str": lambda x: json.dumps(x["compare_tools"], separators=(',', ':'))
        }
        | prompt
        | chat_llm
        | StrOutputParser()
    )

    with get_openai_callback() as cb:
        response = chain.invoke(state)
    state["actual_tokens"] = {
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "total_tokens": cb.total_tokens
    }
    state["report"] = response
    return state


def build_graph():
    wf = StateGraph(ComparisonState)
    wf.add_node("prepare", prepare_data)
    wf.add_node("compare", generate_comparison)
    wf.set_entry_point("prepare")
    wf.add_edge("prepare", "compare")
    wf.add_edge("compare", END)
    return wf.compile()


# Cached for sheet names only (static)
@st.cache_data
def load_sheet_names(_excel_path):
    xls = pd.ExcelFile(_excel_path)
    return xls.sheet_names


# No cache for load_tools to force fresh read every time
def load_tools(_excel_path, _sheet_name):
    df = pd.read_excel(_excel_path, sheet_name=_sheet_name)
    df.columns = df.columns.str.strip()
    return df


graph = build_graph()

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Research & Analysis", page_icon="🔍")

st.title("🔍 Research & Analysis")

# Use session state to track selected sector, sheet and reset tool when they change
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = list(SECTOR_PATHS.keys())[0]
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None
if 'input_choice' not in st.session_state:
    st.session_state.input_choice = None
if 'comparison_result' not in st.session_state:
    st.session_state.comparison_result = None

# Create two columns: left for inputs, right for report
col1, col2 = st.columns([1, 2])  # Left narrower for inputs, right wider for report

with col1:
    st.subheader("⚙️ Options")
    
    with st.container():
        st.markdown("---")
        selected_sector = st.selectbox("Select Sector:", list(SECTOR_PATHS.keys()), index=list(SECTOR_PATHS.keys()).index(st.session_state.selected_sector), key="sector_select")

        if selected_sector != st.session_state.selected_sector:
            st.session_state.selected_sector = selected_sector
            st.session_state.selected_sheet = None
            st.session_state.input_choice = None
            st.session_state.comparison_result = None
            st.rerun()

        excel_path = SECTOR_PATHS.get(selected_sector)

        if excel_path is None:
            st.info("🛠️ Data for the selected sector is in progress. Please check back later.")
            sheet_names = ["In Progress"]
            selected_sheet = st.selectbox("Select stage (sheet) for comparison:", sheet_names, disabled=True, key="sheet_select_disabled")
            tools = ["In Progress"]
            input_choice = st.selectbox("Select input tool to compare:", tools, disabled=True, key="tool_select_disabled")
            compare_button = st.button("Estimate Tokens & Compare", disabled=True, key="compare_disabled")
            st.warning("Functionality will be available once data is ready.")
        else:
            # Load sheet names at top level for Manufacturing (or future sectors with data)
            sheet_names = load_sheet_names(excel_path)

            if st.session_state.selected_sheet is None and sheet_names:
                st.session_state.selected_sheet = sheet_names[0]

            selected_sheet = st.selectbox("Select stage (sheet) for comparison:", sheet_names, index=sheet_names.index(st.session_state.selected_sheet) if st.session_state.selected_sheet in sheet_names else 0, key="sheet_select")

            if selected_sheet != st.session_state.selected_sheet:
                st.session_state.selected_sheet = selected_sheet
                st.session_state.input_choice = None
                st.session_state.comparison_result = None
                st.rerun()

            if selected_sheet and selected_sheet != "In Progress":
                # Load fresh data (no cache)
                df = load_tools(excel_path, selected_sheet)
                tools = df['Tool Name'].dropna().astype(str).unique().tolist()

                if len(tools) == 0:
                    st.warning("No tools found in the selected sheet.")
                else:
                    # Reset input_choice if not in current tools
                    if st.session_state.input_choice not in tools:
                        st.session_state.input_choice = tools[0]

                    # Unique key for widget recreation
                    input_choice = st.selectbox(
                        "Select input tool to compare:", 
                        tools, 
                        index=tools.index(st.session_state.input_choice) if st.session_state.input_choice in tools else 0,
                        key=f"input_tool_select_{selected_sheet.replace(' ', '_').replace('&', 'and')}"
                    )

                    if input_choice != st.session_state.input_choice:
                        st.session_state.input_choice = input_choice

                    relevant_cols = ['Tool Name', 'Tool Type', 'Vendor', 'Region', 'Cost', 
                                     'Number of Integrations', 'Features', 'Efficiency', 'Source']

                    if input_choice:
                        # Ensure the selected tool exists in the current sheet's data
                        tool_row = df[df['Tool Name'] == input_choice]
                        if not tool_row.empty:
                            input_tool = tool_row[relevant_cols].iloc[0].to_dict()
                            compare_tools = df[df['Tool Name'] != input_choice][relevant_cols].to_dict('records')
                            
                            st.markdown("#### 📋 Selected Input Tool")
                            st.markdown("---")

                            display_cols = ['Tool Name', 'Tool Type', 'Vendor', 'Cost']  # Select columns you want
                            filtered_df = pd.DataFrame([input_tool])[display_cols]
                            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                            
                            st.markdown("---")
                            
                            compare_button = st.button("🚀 Compare", key="compare_button", use_container_width=True)
                            
                            if compare_button:
                                with st.spinner("Estimating & Comparing..."):
                                    initial_state = {
                                        "input_tool": input_tool,
                                        "compare_tools": compare_tools,
                                        "report": "",
                                        "estimated_tokens": 0,
                                        "actual_tokens": {},
                                        "estimation_done": False
                                    }
                                    # Manual prepare: Builds prompt once
                                    prep_state = prepare_data(initial_state)
                                    # st.info(f"**Estimated Input Tokens:** ~{prep_state['estimated_tokens']}")
                                    
                                    # Graph invoke: prepare skips due to flag
                                    result = graph.invoke(prep_state)
                                    st.session_state.comparison_result = result
                                    st.rerun()
                        else:
                            st.warning("Selected tool not found in the current sheet data.")
                    
                    with st.expander("📊 Show full data", expanded=False):
                        st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
            else:
                if selected_sheet == "In Progress":
                    st.warning("Sheet data in progress.")
                else:
                    st.warning("No valid sheet selected.")

with col2:
    st.subheader("📊 Comparison Report")
    
    if st.session_state.comparison_result:
        result = st.session_state.comparison_result
        report_text = result.get('report', '')
        
        st.success("✅ Comparison completed successfully!")
        st.markdown("---")
        
        st.download_button(
            label="📥 Download Report as Markdown",
            data=report_text,
            file_name="comparison_report.md",
            mime="text/markdown",
            use_container_width=True
        )
        
        st.markdown("---")
        
        with st.container(border=True):
            st.markdown(report_text)
    else:
        st.info("👆 Select options on the left and click **Compare** to generate the report here.")

        st.markdown("---")


