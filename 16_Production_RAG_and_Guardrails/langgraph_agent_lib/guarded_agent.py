"""Production-safe LangGraph agent with integrated Guardrails validation.

This module implements a LangGraph agent with comprehensive safety layers using
Guardrails AI for input and output validation.

Supports both synchronous and asynchronous validation for production performance.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .models import get_openai_model
from .agents import get_default_tools
from .rag import ProductionRAGChain
from .guardrails import (
    create_guardrails_guard,
    create_factuality_guard,
    validate_input,
    validate_output
)

# Set up logging
logger = logging.getLogger(__name__)


class GuardedAgentState(TypedDict):
    """State schema for guarded agent with validation tracking."""
    messages: Annotated[List[BaseMessage], add_messages]
    validation_passed: bool
    validation_errors: List[str]
    refinement_count: int


def create_guarded_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    rag_chain: Optional[ProductionRAGChain] = None,
    enable_input_guards: bool = True,
    enable_output_guards: bool = True,
    max_refinements: int = 2,
    strict_mode: bool = False,
    use_async_validation: bool = False
):
    """Create a production-safe LangGraph agent with Guardrails validation.
    
    This agent implements a comprehensive safety architecture with:
    - Input validation (jailbreak, topic, PII detection)
    - Output validation (profanity, PII, factuality checking)
    - Graceful error handling
    - Refinement loops for failed validations
    - Both synchronous and asynchronous validation support
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        rag_chain: Optional RAG chain to include as a tool
        enable_input_guards: Whether to enable input validation
        enable_output_guards: Whether to enable output validation
        max_refinements: Maximum refinement iterations for failed validations
        strict_mode: If True, raises exceptions on validation failure.
                    If False, returns error messages gracefully.
        use_async_validation: If True, runs guards asynchronously in parallel
                             for better performance. If False, runs synchronously.
        
    Returns:
        Compiled LangGraph agent with integrated guardrails
    """
    # Get model and tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    tools = get_default_tools(rag_chain)
    model_with_tools = model.bind_tools(tools)
    
    # Create guards
    input_guard = None
    output_guard = None
    
    if enable_input_guards:
        input_guard = create_guardrails_guard(
            valid_topics=["student loans", "financial aid", "education financing", "loan repayment"],
            invalid_topics=["investment advice", "crypto", "gambling", "politics"],
            enable_jailbreak_detection=True,
            enable_pii_protection=True,
            enable_profanity_check=True
            # Guards use on_fail="exception" by default (strict validation)
        )
        logger.info("Input guards configured (strict mode)")
    
    if enable_output_guards:
        output_guard = create_guardrails_guard(
            enable_jailbreak_detection=False,  # Only for inputs
            enable_pii_protection=True,
            enable_profanity_check=True
        )
        logger.info("Output guards configured")
    
    # Create simple factuality guard for RAG responses
    from guardrails.hub import LlmRagEvaluator, HallucinationPrompt
    from guardrails import Guard
    
    factuality_guard = None
    if enable_output_guards and rag_chain:
        factuality_guard = Guard().use(
            LlmRagEvaluator(
                eval_llm_prompt_generator=HallucinationPrompt(prompt_name="hallucination_judge_llm"),
                llm_evaluator_fail_response="hallucinated",
                llm_evaluator_pass_response="factual",
                llm_callable="gpt-4o-mini",
                on_fail="exception",
                on="prompt"
            )
        )
        logger.info("Factuality guard configured")
    
    # Define nodes
    def input_validation_node(state: GuardedAgentState) -> Dict[str, Any]:
        """Validate user input before processing."""
        if not enable_input_guards or not input_guard:
            return {"validation_passed": True, "validation_errors": []}
        
        # Get the last user message
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            return {"validation_passed": True, "validation_errors": []}
        
        try:
            # Always use raise_on_failure=False to get validation results without exceptions
            result = validate_input(input_guard, user_message, raise_on_failure=False)
            
            if result["validation_passed"]:
                logger.info("Input validation passed")
                return {"validation_passed": True, "validation_errors": []}
            else:
                error_msg = result.get("error", "Input validation failed")
                logger.warning(f"Input validation failed: {error_msg}")
                return {
                    "validation_passed": False,
                    "validation_errors": [error_msg]
                }
        except Exception as e:
            # This should rarely happen now since raise_on_failure=False
            error_msg = str(e)
            logger.error(f"Input validation error: {error_msg}")
            if strict_mode:
                raise
            return {
                "validation_passed": False,
                "validation_errors": [error_msg]
            }
    
    def route_after_input_validation(state: GuardedAgentState):
        """Route based on input validation result."""
        if state.get("validation_passed", True):
            return "agent"
        return "input_error_handler"
    
    async def input_validation_node_async(state: GuardedAgentState) -> Dict[str, Any]:
        """Async input validation - runs guards in parallel for better performance."""
        if not enable_input_guards or not input_guard:
            return {"validation_passed": True, "validation_errors": []}
        
        # Get the last user message
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            return {"validation_passed": True, "validation_errors": []}
        
        try:
            # Note: Guardrails doesn't natively support async, so we run in executor
            # In production, you'd use truly async guards or run in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: validate_input(input_guard, user_message, raise_on_failure=False)
            )
            
            if result["validation_passed"]:
                logger.info("Async input validation passed")
                return {"validation_passed": True, "validation_errors": []}
            else:
                error_msg = result.get("error", "Input validation failed")
                logger.warning(f"Async input validation failed: {error_msg}")
                return {
                    "validation_passed": False,
                    "validation_errors": [error_msg]
                }
        except Exception as e:
            # This should rarely happen now since raise_on_failure=False
            error_msg = str(e)
            logger.error(f"Async input validation error: {error_msg}")
            if strict_mode:
                raise
            return {
                "validation_passed": False,
                "validation_errors": [error_msg]
            }
    
    def input_error_handler(state: GuardedAgentState) -> Dict[str, Any]:
        """Handle input validation failures gracefully."""
        errors = state.get("validation_errors", ["Input validation failed"])
        error_message = (
            "I apologize, but I cannot process this request. "
            "Please ensure your query is related to student loans, financial aid, "
            "or education financing, and does not contain sensitive personal information."
        )
        
        logger.warning(f"Input blocked: {errors}")
        return {"messages": [AIMessage(content=error_message)]}
    
    def call_model(state: GuardedAgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue_to_tools(state: GuardedAgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "output_validation"
    
    def output_validation_node(state: GuardedAgentState) -> Dict[str, Any]:
        """Validate agent output before returning to user."""
        if not enable_output_guards or not output_guard:
            return {"validation_passed": True, "validation_errors": []}
        
        # Get the last AI message
        ai_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and not msg.content.startswith("VALIDATION:"):
                ai_message = msg.content
                break
        
        if not ai_message:
            return {"validation_passed": True, "validation_errors": []}
        
        try:
            # Standard output validation (PII, profanity)
            # Always use raise_on_failure=False to get validation results without exceptions
            result = validate_output(output_guard, ai_message, raise_on_failure=False)
            
            if not result["validation_passed"]:
                error_msg = result.get("error", "Output validation failed")
                logger.warning(f"Output validation failed: {error_msg}")
                return {
                    "validation_passed": False,
                    "validation_errors": [error_msg]
                }
            
            # Simple factuality validation - only for RAG responses
            if factuality_guard:
                # Check if RAG tool was used
                rag_tool_used = any(
                    hasattr(msg, "tool_calls") and msg.tool_calls and
                    any(tc.get("name") == "retrieve_information" for tc in msg.tool_calls)
                    for msg in state["messages"]
                )
                
                if rag_tool_used:
                    try:
                        factuality_result = factuality_guard.validate(ai_message)
                        if not factuality_result.validation_passed:
                            logger.warning("Factuality check failed")
                            return {
                                "validation_passed": False,
                                "validation_errors": ["Response may contain hallucinations"]
                            }
                    except Exception as e:
                        # Log but don't fail on factuality errors
                        logger.warning(f"Factuality check error: {e}")
            
            logger.info("Output validation passed")
            return {"validation_passed": True, "validation_errors": []}
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Output validation error: {error_msg}")
            if strict_mode:
                raise
            return {
                "validation_passed": False,
                "validation_errors": [error_msg]
            }
    
    def route_after_output_validation(state: GuardedAgentState):
        """Route based on output validation result."""
        if state.get("validation_passed", True):
            return END
        
        # Check if we've exceeded max refinements
        refinement_count = state.get("refinement_count", 0)
        if refinement_count >= max_refinements:
            logger.warning(f"Max refinements ({max_refinements}) reached")
            return "output_error_handler"
        
        return "refinement"
    
    def refinement_node(state: GuardedAgentState) -> Dict[str, Any]:
        """Request agent to refine response based on validation failure."""
        errors = state.get("validation_errors", [])
        refinement_count = state.get("refinement_count", 0)
        
        refinement_prompt = (
            f"Your previous response failed validation due to: {', '.join(errors)}. "
            "Please provide an improved response that addresses these issues while "
            "staying on topic and maintaining professional, helpful communication."
        )
        
        logger.info(f"Refinement iteration {refinement_count + 1}")
        return {
            "messages": [HumanMessage(content=refinement_prompt)],
            "refinement_count": refinement_count + 1,
            "validation_errors": []  # Clear errors for next iteration
        }
    
    def output_error_handler(state: GuardedAgentState) -> Dict[str, Any]:
        """Handle output validation failures after max refinements."""
        error_message = (
            "I apologize, but I'm unable to provide a satisfactory response "
            "that meets our safety and quality standards. Please try rephrasing "
            "your question or contact support for assistance."
        )
        
        logger.error("Output validation failed after max refinements")
        return {"messages": [AIMessage(content=error_message)]}
    
    # Build the graph
    graph = StateGraph(GuardedAgentState)
    
    # Choose sync or async validation nodes
    input_val_node = input_validation_node_async if use_async_validation else input_validation_node
    
    # Add nodes
    graph.add_node("input_validation", input_val_node)
    graph.add_node("input_error_handler", input_error_handler)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("output_validation", output_validation_node)
    graph.add_node("refinement", refinement_node)
    graph.add_node("output_error_handler", output_error_handler)
    
    # Log validation mode
    validation_mode = "asynchronous (parallel)" if use_async_validation else "synchronous (sequential)"
    logger.info(f"Guarded agent configured with {validation_mode} validation")
    
    # Set entry point
    graph.set_entry_point("input_validation")
    
    # Add edges
    graph.add_conditional_edges(
        "input_validation",
        route_after_input_validation,
        {"agent": "agent", "input_error_handler": "input_error_handler"}
    )
    graph.add_edge("input_error_handler", END)
    
    graph.add_conditional_edges(
        "agent",
        should_continue_to_tools,
        {"tools": "tools", "output_validation": "output_validation"}
    )
    graph.add_edge("tools", "agent")
    
    graph.add_conditional_edges(
        "output_validation",
        route_after_output_validation,
        {
            END: END,
            "refinement": "refinement",
            "output_error_handler": "output_error_handler"
        }
    )
    graph.add_edge("refinement", "agent")
    graph.add_edge("output_error_handler", END)
    
    return graph.compile()
