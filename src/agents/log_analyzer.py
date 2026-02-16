"""
Log Analyzer Agent with Kubernetes Actions (Chapter 9)
AI agent for analyzing logs and managing Kubernetes pods
"""
import time

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from ..models import get_model
from ..tools import get_all_tools, is_write_tool
from ..utils.response import extract_response_text
from ..utils.tracing import JsonlTraceWriter, TraceSpan, new_trace_id, trace_config_from_env
from ..config import Config


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class LogAnalyzerAgent:
    """
    AI Logging Agent with Kubernetes Management Capabilities
    
    Capabilities:
    - Read and analyze application logs from Kubernetes pods
    - Detect critical issues (OutOfMemoryError, CrashLoopBackOff, etc.)
    - Automatically restart failed pods for P1 issues
    - Check pod status and retrieve pod logs
    - Scale deployments when needed
    - Maintain conversation history
    """
    
    def __init__(self, model_provider: str | None = None, model_name: str | None = None):
        """Initialize the agent"""
        # Initialize model
        self.model = get_model(provider=model_provider, model_name=model_name)
        self.llm = self.model.get_llm()
        
        # Get all tools (log readers + K8s actions)
        self.tools = get_all_tools()
        
        # Bind tools to model
        self.llm_with_tools = self.model.get_llm_with_tools(self.tools)

        # Pending write action (requires explicit user approval)
        self._pending_action: dict | None = None
        
        # Tracing
        self.last_trace_id: str | None = None
        self._trace_writer = None
        try:
            cfg = trace_config_from_env(default_dir=Config.TRACE_DIR)
            if cfg.enabled:
                self._trace_writer = JsonlTraceWriter(cfg)
        except Exception:
            self._trace_writer = None
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", Config.get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
    
    def process_query(self, user_input: str, chat_history: list = None) -> str:
        """
        Process a user query and return the response.
        
        Args:
            user_input: User's question or command
            chat_history: List of previous messages (HumanMessage, AIMessage)
        
        Returns:
            String containing the agent's response
        """
        if chat_history is None:
            chat_history = []
        
        trace_id = new_trace_id()
        self.last_trace_id = trace_id
        tw = self._trace_writer
        if tw:
            tw.emit({
                "trace_id": trace_id,
                "event": "turn.start",
                "provider": Config.LLM_PROVIDER,
                "model": Config.get_active_model_name(),
                "chat_history_len": len(chat_history),
                "user_input": user_input,
            })
        
        try:
            # If there is a pending write action, require an explicit confirmation.
            if self._pending_action is not None:
                decision = user_input.strip().lower()
                if decision in {"yes", "y", "approve", "proceed", "do it", "run it", "ok"} or decision.startswith("approve "):
                    tool = self._pending_action["tool"]
                    args = self._pending_action["args"]
                    self._pending_action = None
                    try:
                        if tw:
                            tw.emit({"trace_id": trace_id, "event": "approval.accept", "tool": tool.name, "args": args})
                        with TraceSpan(tw, trace_id, "tool.invoke", {"tool": getattr(tool, "name", "<unknown>") , "args": args}) if tw else _NullContext():
                            result = tool.invoke(args)
                        return (
                            f"Approved. Executed `{tool.name}`.\n\n"
                            f"Result:\n{result}"
                        )
                    except Exception as e:
                        if tw:
                            tw.emit({"trace_id": trace_id, "event": "approval.exec_error", "tool": tool.name, "error": str(e)})
                        return f"Approved, but execution failed: {e}"

                if decision in {"no", "n", "cancel", "stop"}:
                    pending_name = self._pending_action["tool"].name
                    self._pending_action = None
                    if tw:
                        tw.emit({"trace_id": trace_id, "event": "approval.deny", "tool": pending_name})
                    return f"Cancelled. I will not run `{pending_name}`."

                pending_name = self._pending_action["tool"].name
                pending_args = self._pending_action["args"]
                return (
                    f"Approval required before I can run `{pending_name}` with args {pending_args}.\n"
                    "Reply `yes` to approve or `no` to cancel."
                )

            # Format messages for the prompt
            messages = self.prompt.format_messages(
                chat_history=chat_history,
                input=user_input
            )
            
            # Get response from LLM with tools
            t0 = time.perf_counter()
            response = None
            try:
                response = self.llm_with_tools.invoke(messages)
            finally:
                if tw:
                    tw.emit({
                        "trace_id": trace_id,
                        "event": "llm.invoke",
                        "duration_ms": (time.perf_counter() - t0) * 1000.0,
                        "has_tool_calls": bool(getattr(response, 'tool_calls', None)),
                        "usage": getattr(response, 'usage_metadata', None) or getattr(response, 'response_metadata', None),
                    })
            
            # Check if model wants to use tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                final = self._handle_tool_calls(response, user_input, chat_history, trace_id=trace_id)
                if tw:
                    tw.emit({"trace_id": trace_id, "event": "turn.end", "final_len": len(final)})
                if not (final or "").strip():
                    trace_hint = f" Trace ID: {trace_id}" if (tw and trace_id) else ""
                    return (
                        "I got an empty response from the model after tool execution. "
                        f"Provider={Config.LLM_PROVIDER}, Model={Config.get_active_model_name()}."
                        + trace_hint
                    )
                return final
            else:
                # Direct response without tools
                final = extract_response_text(response)
                if tw:
                    tw.emit({"trace_id": trace_id, "event": "turn.end", "final_len": len(final)})
                if not (final or "").strip():
                    trace_hint = f" Trace ID: {trace_id}" if (tw and trace_id) else ""
                    return (
                        "I got an empty response from the model (no text content). "
                        f"Provider={Config.LLM_PROVIDER}, Model={Config.get_active_model_name()}."
                        + trace_hint
                    )
                return final
        
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"\n{error_msg}")
            import traceback
            traceback.print_exc()
            if tw:
                tw.emit({"trace_id": trace_id, "event": "turn.error", "error": str(e)})
            return error_msg
    
    def _handle_tool_calls(self, response, user_input: str, chat_history: list, *, trace_id: str | None = None) -> str:
        """
        Handle tool calls from the model with iterative execution.
        
        Args:
            response: LLM response containing tool calls
            user_input: Original user input
            chat_history: Conversation history
        
        Returns:
            Final response after executing tools
        """
        from langchain_core.messages import AIMessage, ToolMessage

        tw = self._trace_writer
        
        # Keep track of all messages for the agent loop
        messages = self.prompt.format_messages(
            chat_history=chat_history,
            input=user_input
        )
        
        # Agent loop: continue until no more tool calls
        max_iterations = getattr(Config, "MAX_ITERATIONS", 5)
        iteration = 0
        current_response = response
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check if there are tool calls
            if not (hasattr(current_response, 'tool_calls') and current_response.tool_calls):
                # No more tool calls, return the final response
                return extract_response_text(current_response)
            
            # Execute each tool call
            tool_messages = []
            for tool_call in current_response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                if tw and trace_id:
                    tw.emit({"trace_id": trace_id, "event": "tool.request", "tool": tool_name, "args": tool_args})
                
                # Find and execute the tool
                tool_func = None
                for tool in self.tools:
                    if tool.name == tool_name:
                        tool_func = tool
                        break
                
                if tool_func:
                    # Enforce explicit approval for write tools.
                    if is_write_tool(tool_name):
                        self._pending_action = {
                            "tool": tool_func,
                            "args": tool_args,
                        }
                        if tw and trace_id:
                            tw.emit({"trace_id": trace_id, "event": "tool.requires_approval", "tool": tool_name, "args": tool_args})
                        return (
                            f"I recommend running `{tool_name}` with args {tool_args}, but it requires approval.\n"
                            "Would you like me to proceed? (yes/no)"
                        )
                    try:
                        t_tool0 = time.perf_counter()
                        result = tool_func.invoke(tool_args)
                        if tw and trace_id:
                            tw.emit({
                                "trace_id": trace_id,
                                "event": "tool.result",
                                "tool": tool_name,
                                "duration_ms": (time.perf_counter() - t_tool0) * 1000.0,
                                "result_type": type(result).__name__,
                                "result_len": len(str(result)),
                            })
                        tool_messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call['id']
                            )
                        )
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        if tw and trace_id:
                            tw.emit({"trace_id": trace_id, "event": "tool.error", "tool": tool_name, "error": str(e)})
                        tool_messages.append(
                            ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_call['id']
                            )
                        )
            
            # Add AI response and tool results to messages
            messages.append(AIMessage(content=current_response.content, tool_calls=current_response.tool_calls))
            messages.extend(tool_messages)
            
            # Get next response from LLM (might call more tools or finish)
            t1 = time.perf_counter()
            current_response = self.llm_with_tools.invoke(messages)
            if tw and trace_id:
                tw.emit({
                    "trace_id": trace_id,
                    "event": "llm.invoke",
                    "duration_ms": (time.perf_counter() - t1) * 1000.0,
                    "has_tool_calls": bool(getattr(current_response, 'tool_calls', None)),
                    "usage": getattr(current_response, 'usage_metadata', None) or getattr(current_response, 'response_metadata', None),
                })


        # If we hit max iterations and the model is still requesting tools, force a final answer
        # using the base LLM (tools disabled) so the UI doesn't look "stuck".
        if hasattr(current_response, 'tool_calls') and current_response.tool_calls:
            if tw and trace_id:
                tw.emit({
                    "trace_id": trace_id,
                    "event": "tool_loop.max_iterations_hit",
                    "max_iterations": max_iterations,
                    "remaining_tool_calls": len(current_response.tool_calls),
                })

            messages.append(
                HumanMessage(
                    content=(
                        "Stop calling tools now. Provide your best incident summary based only on the tool results already retrieved. "
                        "If the evidence is insufficient, ask ONE specific clarifying question instead of calling more tools."
                    )
                )
            )
            t_force = time.perf_counter()
            forced = self.llm.invoke(messages)
            if tw and trace_id:
                tw.emit({
                    "trace_id": trace_id,
                    "event": "llm.invoke.force_final",
                    "duration_ms": (time.perf_counter() - t_force) * 1000.0,
                    "usage": getattr(forced, 'usage_metadata', None) or getattr(forced, 'response_metadata', None),
                })
            forced_text = extract_response_text(forced)
            if (forced_text or "").strip():
                return forced_text

        # If we hit max iterations, return what we have
        final = extract_response_text(current_response)
        if not (final or "").strip():
            trace_hint = f" Trace ID: {trace_id}" if (tw and trace_id) else ""
            return (
                "I got an empty response from the model at the end of the tool loop. "
                f"Provider={Config.LLM_PROVIDER}, Model={Config.get_active_model_name()}."
                + trace_hint
            )
        return final


    def clear_history(self):
        """Clear any pending action. Streamlit history is managed externally."""
        self._pending_action = None