"""AI Ops Kubernetes Assistant."""
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents import LogAnalyzerAgent
from src.config import Config


# Page configuration
st.set_page_config(
    page_title="AI Ops K8s Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'agent' not in st.session_state:
        try:
            Config.validate()
            st.session_state.agent = LogAnalyzerAgent()
        except ValueError as e:
            st.error(f"Configuration error: {e}")
            st.stop()

    if 'model_provider' not in st.session_state:
        st.session_state.model_provider = Config.LLM_PROVIDER
    if 'model_name' not in st.session_state:
        st.session_state.model_name = Config.get_active_model_name()


def display_sidebar():
    """Display sidebar with information and controls"""
    with st.sidebar:
        st.title("AI Ops Agent")
        st.markdown("---")

        st.subheader("Model")
        provider = st.selectbox(
            "Provider",
            options=["gemini", "openrouter"],
            index=0 if st.session_state.model_provider == "gemini" else 1,
        )

        openrouter_presets = [
            "qwen/qwen3-vl-235b-a22b-thinking",
            "stepfun/step-3.5-flash:free",
            "openai/gpt-oss-120b:free"
        ]

        model_name = None
        if provider == "gemini":
            model_name = st.text_input("Gemini model", value=Config.GEMINI_MODEL)
        else:
            model_name = st.selectbox("OpenRouter model", options=openrouter_presets, index=0)

        if st.button("Apply model", use_container_width=True):
            st.session_state.model_provider = provider
            st.session_state.model_name = model_name
            # Recreate the agent with the selected model.
            st.session_state.agent = LogAnalyzerAgent(model_provider=provider, model_name=model_name)
            st.rerun()
        
        st.subheader("About")
        st.markdown("""
        AI-powered Kubernetes diagnostics and incident response:
        - Cluster and namespace diagnostics
        - Pod/deployment/service investigation
        - Pod logs from the cluster API
        - Event and resource pressure analysis
        - Safe action execution with approval
        """)
        
        st.markdown("---")
        
        st.subheader("Available Tools")
        st.markdown("""
        **Kubernetes Diagnostics (Read-only):**
        - `k8s_current_context`, `k8s_list_contexts`, `k8s_cluster_info`
        - `k8s_list_namespaces`, `k8s_list_nodes`, `k8s_top_nodes`
        - `k8s_list_pods`, `k8s_find_pods`, `k8s_describe_pod`
        - `k8s_get_pod_logs`, `k8s_get_events`, `k8s_get_crashloop_pods`
        - `k8s_list_deployments`, `k8s_describe_deployment`
        - `k8s_list_services`, `k8s_list_ingresses`, `k8s_list_hpa`
        
        **Kubernetes Actions:**
        - `restart_kubernetes_pod` - Restart failed pod
          - 🔒 Always asks for approval
                    - ⚡ Use for crash-recovery scenarios
        """)
        
        st.markdown("---")
        
        st.subheader("Example Questions")
        st.markdown("""
        - "Show current context and list namespaces"
        - "Find nexus pods and describe the unhealthy one"
        - "Get pod logs for last 30 minutes"
        - "Show recent warning/error events in production"
        - "Which pods are in CrashLoopBackOff?"
        """)
        
        st.markdown("---")
        
        st.subheader("Severity Levels")
        st.markdown("""
        - **P1**: OOM, pod crashes
        - **P2**: Errors, degradation
        - **P3**: Warnings
        - **Info**: Normal operations
        """)
        
        st.markdown("---")
        
        st.subheader("How It Works")
        st.markdown("""
        1. **Discovery**: AI queries cluster state (pods, events, workloads)
        2. **Diagnosis**: Correlates status, logs, and resource signals
        3. **Recommendation**: Suggests actions (e.g., pod restart)
        4. **Confirmation**: Asks for your approval
        5. **Execution**: Performs action after you confirm
        """)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # System info
        st.markdown("---")
        st.caption(f"Provider: {st.session_state.model_provider}")
        st.caption(f"Model: {st.session_state.model_name}")
        st.caption(f"Temperature: {Config.TEMPERATURE}")
        st.caption(f"K8s Context: {Config.K8S_CONTEXT or '(active kubectl context)'}")
        st.caption(f"K8s Namespace: {Config.K8S_DEFAULT_NAMESPACE or '(auto/current context)'}")

        st.markdown("---")
        st.subheader("Tracing")
        st.caption(f"Enabled: {Config.TRACE_ENABLED}")
        st.caption(f"Dir: {Config.TRACE_DIR}")
        last_trace_id = None
        try:
            last_trace_id = getattr(st.session_state.get('agent', None), 'last_trace_id', None)
        except Exception:
            last_trace_id = None
        if last_trace_id:
            st.code(last_trace_id)


def display_chat_messages():
    """Display all chat messages from history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def convert_to_langchain_messages(messages):
    """Convert Streamlit messages to LangChain message format"""
    langchain_messages = []
    for msg in messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
    return langchain_messages


def main():
    """Main application logic"""
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    st.markdown("Investigate Kubernetes incidents with intelligent cluster diagnostics")
    
    # Info banner
    st.info("""
    💡 **Tip**: Ask me to inspect a namespace/service (pods, events, logs, resources).
    I will diagnose first, recommend remediation, and ask confirmation before any write action.
    """)
    
    # Display chat messages
    display_chat_messages()
    
    # Chat input
    if prompt := st.chat_input("Ask about cluster health, pods, events, or remediation..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Inspecting cluster state and diagnostics..."):
                # Convert chat history to LangChain format
                chat_history = convert_to_langchain_messages(
                    st.session_state.messages[:-1]  # Exclude the current message
                )
                
                # Get response from agent
                response = st.session_state.agent.process_query(
                    user_input=prompt,
                    chat_history=chat_history
                )
                
                # Display response (avoid silent blank)
                if not (response or "").strip():
                    st.warning("Agent returned an empty response. Check model/provider settings or enable tracing.")
                else:
                    st.markdown(response)

                # Surface trace id for debugging
                trace_id = getattr(st.session_state.agent, 'last_trace_id', None)
                if trace_id and Config.TRACE_ENABLED:
                    st.caption(f"Trace ID: {trace_id}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()