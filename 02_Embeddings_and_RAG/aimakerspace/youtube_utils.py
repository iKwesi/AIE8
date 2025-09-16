"""
YouTube utilities for extracting and processing video content.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import requests


class YouTubeTranscriptLoader:
    """
    Loads and processes YouTube video transcripts.
    """
    
    def __init__(self):
        self.video_id = None
        self.video_info = {}
    
    def extract_video_id(self, url: str) -> str:
        """
        Extract video ID from various YouTube URL formats.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID string
        """
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract video ID from URL: {url}")
    
    def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """
        Get basic video information (mock implementation for demo).
        In production, you'd use YouTube Data API.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with video metadata
        """
        # Mock video info - in production, use YouTube Data API
        return {
            "video_id": video_id,
            "title": f"Video {video_id}",
            "channel": "Sample Channel",
            "duration": "10:30",
            "upload_date": datetime.now().isoformat(),
            "description": "Sample video description",
            "view_count": 1000,
            "like_count": 50,
            "language": "en"
        }
    
    def get_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Get video transcript (mock implementation for demo).
        In production, you'd use youtube-transcript-api.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of transcript segments with timestamps
        """
        # Mock transcript - in production, use youtube-transcript-api
        mock_transcript = [
            {
                "text": "Welcome to this video about startup advice and entrepreneurship.",
                "start": 0.0,
                "duration": 3.5
            },
            {
                "text": "Today we'll discuss the key factors that determine startup success.",
                "start": 3.5,
                "duration": 4.0
            },
            {
                "text": "First, let's talk about product market fit and why it's crucial.",
                "start": 7.5,
                "duration": 4.2
            },
            {
                "text": "Product market fit means customers are buying your product as fast as you can make it.",
                "start": 11.7,
                "duration": 5.1
            },
            {
                "text": "Without product market fit, even the best team will struggle to succeed.",
                "start": 16.8,
                "duration": 4.3
            },
            {
                "text": "Next, we need to consider the founding team and their experience.",
                "start": 21.1,
                "duration": 4.0
            },
            {
                "text": "A strong founding team with complementary skills is essential for startup success.",
                "start": 25.1,
                "duration": 5.2
            },
            {
                "text": "The team should include technical expertise, business acumen, and domain knowledge.",
                "start": 30.3,
                "duration": 5.5
            },
            {
                "text": "Market timing is another critical factor that many entrepreneurs overlook.",
                "start": 35.8,
                "duration": 4.7
            },
            {
                "text": "Even great products can fail if they're introduced too early or too late.",
                "start": 40.5,
                "duration": 4.8
            }
        ]
        
        return mock_transcript
    
    def load_video_content(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """
        Load complete video content including transcript and metadata.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Tuple of (full_transcript_text, metadata)
        """
        try:
            # Extract video ID
            video_id = self.extract_video_id(url)
            self.video_id = video_id
            
            # Get video info
            video_info = self.get_video_info(video_id)
            
            # Get transcript
            transcript_segments = self.get_transcript(video_id)
            
            # Combine transcript text
            full_transcript = " ".join([segment["text"] for segment in transcript_segments])
            
            # Create comprehensive metadata
            metadata = {
                **video_info,
                "source": url,
                "source_type": "youtube",
                "transcript_segments": transcript_segments,
                "total_segments": len(transcript_segments),
                "transcript_length": len(full_transcript),
                "loader": "YouTubeTranscriptLoader",
                "processed_date": datetime.now().isoformat()
            }
            
            self.video_info = metadata
            
            return full_transcript, metadata
            
        except Exception as e:
            raise Exception(f"Failed to load YouTube video content: {str(e)}")


class YouTubeTextSplitter:
    """
    Splits YouTube transcript text into chunks while preserving timestamp information.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, preserve_timestamps: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_timestamps = preserve_timestamps
    
    def split_by_segments(self, transcript_segments: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split transcript by combining segments into chunks.
        
        Args:
            transcript_segments: List of transcript segments with timestamps
            metadata: Video metadata
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        current_chunk = ""
        current_segments = []
        chunk_id = 0
        
        for segment in transcript_segments:
            segment_text = segment["text"]
            
            # Check if adding this segment would exceed chunk size
            if len(current_chunk + " " + segment_text) > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk_metadata = self._create_chunk_metadata(
                    metadata, current_segments, chunk_id
                )
                
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": chunk_metadata
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + segment_text if overlap_text else segment_text
                current_segments = [segment]
                chunk_id += 1
            else:
                # Add segment to current chunk
                current_chunk += " " + segment_text if current_chunk else segment_text
                current_segments.append(segment)
        
        # Add final chunk if there's content
        if current_chunk:
            chunk_metadata = self._create_chunk_metadata(
                metadata, current_segments, chunk_id
            )
            
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def split_by_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split transcript text into chunks (fallback method).
        
        Args:
            text: Full transcript text
            metadata: Video metadata
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            chunk_metadata = {
                **metadata,
                "chunk_id": chunk_id,
                "chunk_start": i,
                "chunk_end": min(i + self.chunk_size, len(text)),
                "chunk_size": len(chunk_text),
                "timestamp_start": None,
                "timestamp_end": None,
                "segments_included": None
            }
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            
            chunk_id += 1
        
        return chunks
    
    def split(self, transcript_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main split method that chooses the best approach.
        
        Args:
            transcript_text: Full transcript text
            metadata: Video metadata
            
        Returns:
            List of chunk dictionaries
        """
        # If we have transcript segments and want to preserve timestamps
        if self.preserve_timestamps and "transcript_segments" in metadata:
            return self.split_by_segments(metadata["transcript_segments"], metadata)
        else:
            return self.split_by_text(transcript_text, metadata)
    
    def _create_chunk_metadata(self, base_metadata: Dict[str, Any], segments: List[Dict[str, Any]], chunk_id: int) -> Dict[str, Any]:
        """Create metadata for a chunk based on its segments."""
        if not segments:
            return {**base_metadata, "chunk_id": chunk_id}
        
        start_time = segments[0]["start"]
        end_time = segments[-1]["start"] + segments[-1]["duration"]
        
        chunk_metadata = {
            **base_metadata,
            "chunk_id": chunk_id,
            "timestamp_start": start_time,
            "timestamp_end": end_time,
            "duration": end_time - start_time,
            "segments_included": len(segments),
            "segment_range": f"{start_time:.1f}s - {end_time:.1f}s"
        }
        
        # Remove the full transcript_segments to avoid duplication
        if "transcript_segments" in chunk_metadata:
            del chunk_metadata["transcript_segments"]
        
        return chunk_metadata
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]


# Production note: To use real YouTube functionality, install these packages:
# pip install youtube-transcript-api google-api-python-client

class ProductionYouTubeLoader:
    """
    Production-ready YouTube loader (requires additional dependencies).
    This is a template for real implementation.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        # In production: from youtube_transcript_api import YouTubeTranscriptApi
        # In production: from googleapiclient.discovery import build
    
    def get_real_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Get real transcript using youtube-transcript-api.
        """
        # Production implementation:
        # try:
        #     transcript = YouTubeTranscriptApi.get_transcript(video_id)
        #     return transcript
        # except Exception as e:
        #     raise Exception(f"Could not retrieve transcript: {str(e)}")
        
        raise NotImplementedError("Install youtube-transcript-api for production use")
    
    def get_real_video_info(self, video_id: str) -> Dict[str, Any]:
        """
        Get real video info using YouTube Data API.
        """
        # Production implementation:
        # if not self.api_key:
        #     raise ValueError("YouTube Data API key required")
        # 
        # youtube = build('youtube', 'v3', developerKey=self.api_key)
        # request = youtube.videos().list(part='snippet,statistics', id=video_id)
        # response = request.execute()
        # 
        # if not response['items']:
        #     raise ValueError(f"Video not found: {video_id}")
        # 
        # video = response['items'][0]
        # return {
        #     'title': video['snippet']['title'],
        #     'channel': video['snippet']['channelTitle'],
        #     'description': video['snippet']['description'],
        #     'upload_date': video['snippet']['publishedAt'],
        #     'view_count': video['statistics'].get('viewCount', 0),
        #     'like_count': video['statistics'].get('likeCount', 0),
        #     'comment_count': video['statistics'].get('commentCount', 0)
        # }
        
        raise NotImplementedError("Install google-api-python-client and provide API key for production use")
