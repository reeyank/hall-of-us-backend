from typing import Optional, Dict, Any, List
from loguru import logger


class ChatContext:
    """Shared context for chat-based image processing operations"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset all context state"""
        self.image_url: Optional[str] = None
        self.image_metadata: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_tags: List[str] = []
        self.current_caption: Optional[str] = None
        self.additional_context: Dict[str, Any] = {}
        logger.info("Chat context reset")

    def set_image(self, image_url: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Set the current image for all operations"""
        self.image_url = image_url
        if metadata:
            self.image_metadata = metadata
        logger.info(f"Image set in context: {image_url}")

    def add_to_conversation(self, operation: str, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> None:
        """Add an operation to the conversation history"""
        entry = {
            "operation": operation,
            "request": request_data,
            "response": response_data,
            "timestamp": self._get_current_timestamp()
        }
        self.conversation_history.append(entry)
        logger.debug(f"Added to conversation history: {operation}")

    def update_tags(self, tags: List[str]) -> None:
        """Update the current tags in context"""
        self.current_tags = tags
        logger.debug(f"Updated tags in context: {tags}")

    def update_caption(self, caption: str) -> None:
        """Update the current caption in context"""
        self.current_caption = caption
        logger.debug(f"Updated caption in context: {caption}")

    def set_additional_context(self, context: Dict[str, Any]) -> None:
        """Set additional context information"""
        self.additional_context = context
        logger.debug("Updated additional context")

    def get_context_summary(self) -> str:
        """Get a formatted context summary for prompts"""
        context_parts = []

        if self.current_tags:
            context_parts.append(f"Current tags: {', '.join(self.current_tags)}")

        if self.current_caption:
            context_parts.append(f"Current caption: {self.current_caption}")

        if self.conversation_history:
            recent_ops = [entry["operation"] for entry in self.conversation_history[-3:]]
            context_parts.append(f"Recent operations: {', '.join(recent_ops)}")

        if self.additional_context:
            for key, value in self.additional_context.items():
                if value:
                    context_parts.append(f"{key}: {value}")

        return "\n".join(context_parts) if context_parts else ""

    def has_image(self) -> bool:
        """Check if an image is set in the context"""
        return self.image_url is not None

    def get_image_url(self) -> Optional[str]:
        """Get the current image URL"""
        return self.image_url

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat()