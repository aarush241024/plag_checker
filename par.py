import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

class ParaphraseDetector:
    def __init__(self):
        # Initialize the model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.similarity_threshold = 0.75  # Adjustable threshold
        
    def detect_paraphrases(self, text1: str, text2: str) -> Dict:
        """
        Detect if two texts are paraphrases of each other
        """
        try:
            # Encode the texts
            with torch.no_grad():
                embedding1 = self.model.encode(text1, convert_to_tensor=True)
                embedding2 = self.model.encode(text2, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            
            # Determine if texts are paraphrases
            is_paraphrase = similarity >= self.similarity_threshold
            
            return {
                'similarity_score': round(similarity * 100, 2),
                'is_paraphrase': is_paraphrase,
                'confidence': self._calculate_confidence(similarity)
            }
        
        except Exception as e:
            st.error(f"Error in paraphrase detection: {str(e)}")
            return {
                'similarity_score': 0,
                'is_paraphrase': False,
                'confidence': 0
            }
    
    def detect_paraphrases_in_list(self, text: str, text_list: List[str]) -> List[Dict]:
        """
        Find paraphrases of a text in a list of texts
        """
        results = []
        try:
            # Encode the main text
            with torch.no_grad():
                main_embedding = self.model.encode(text, convert_to_tensor=True)
                
                # Encode all comparison texts
                comparison_embeddings = self.model.encode(text_list, convert_to_tensor=True)
            
            # Calculate similarities
            similarities = util.pytorch_cos_sim(main_embedding, comparison_embeddings)[0]
            
            # Process results
            for idx, similarity in enumerate(similarities):
                similarity_score = similarity.item()
                results.append({
                    'text': text_list[idx],
                    'similarity_score': round(similarity_score * 100, 2),
                    'is_paraphrase': similarity_score >= self.similarity_threshold,
                    'confidence': self._calculate_confidence(similarity_score)
                })
            
        except Exception as e:
            st.error(f"Error in batch paraphrase detection: {str(e)}")
        
        return results
    
    def _calculate_confidence(self, similarity: float) -> str:
        """
        Calculate confidence level based on similarity score
        """
        if similarity >= 0.9:
            return "Very High"
        elif similarity >= 0.8:
            return "High"
        elif similarity >= 0.7:
            return "Moderate"
        elif similarity >= 0.6:
            return "Low"
        else:
            return "Very Low"

def create_similarity_gauge(similarity_score: float) -> go.Figure:
    """Create a gauge chart for similarity score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=similarity_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 70], 'color': "gray"},
                {'range': [70, 80], 'color': "lightgreen"},
                {'range': [80, 90], 'color': "green"},
                {'range': [90, 100], 'color': "darkgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        },
        title={'text': "Similarity Score"}
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    st.set_page_config(
        page_title="Paraphrase Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Advanced Paraphrase Detector")
    st.markdown("Using **MiniLM** model for semantic similarity analysis")
    
    # Initialize the detector
    @st.cache_resource
    def load_detector():
        return ParaphraseDetector()
    
    detector = load_detector()
    
    # Sidebar settings
    st.sidebar.title("⚙️ Settings")
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Minimum similarity score to consider texts as paraphrases"
    )
    detector.similarity_threshold = similarity_threshold
    
    # Main content
    tabs = st.tabs(["Compare Two Texts", "Batch Comparison"])
    
    # Two Text Comparison
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Text")
            text1 = st.text_area(
                "Enter the first text",
                height=200,
                placeholder="Enter the original text here..."
            )
        
        with col2:
            st.subheader("Comparison Text")
            text2 = st.text_area(
                "Enter the second text",
                height=200,
                placeholder="Enter the text to compare..."
            )
        
        if st.button("Compare Texts", key="compare_two"):
            if text1 and text2:
                with st.spinner("Analyzing texts..."):
                    result = detector.detect_paraphrases(text1, text2)
                
                # Display results
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.plotly_chart(create_similarity_gauge(result['similarity_score']), use_container_width=True)
                
                with col2:
                    st.metric("Similarity Score", f"{result['similarity_score']}%")
                    st.metric("Confidence Level", result['confidence'])
                
                with col3:
                    if result['is_paraphrase']:
                        st.success("✅ PARAPHRASE DETECTED")
                    else:
                        st.warning("❌ NOT A PARAPHRASE")
            else:
                st.warning("Please enter both texts to compare")
    
    # Batch Comparison
    with tabs[1]:
        st.subheader("Batch Text Comparison")
        reference_text = st.text_area(
            "Enter the reference text",
            height=100,
            placeholder="Enter the text to compare against..."
        )
        
        comparison_texts = st.text_area(
            "Enter comparison texts (one per line)",
            height=200,
            placeholder="Enter multiple texts, one per line..."
        )
        
        if st.button("Compare All", key="compare_batch"):
            if reference_text and comparison_texts:
                text_list = [text.strip() for text in comparison_texts.split('\n') if text.strip()]
                
                if text_list:
                    with st.spinner("Analyzing texts..."):
                        results = detector.detect_paraphrases_in_list(reference_text, text_list)
                    
                    # Display results
                    for idx, result in enumerate(results, 1):
                        with st.expander(f"Text {idx} - Similarity: {result['similarity_score']}%"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.text_area("Compared Text", result['text'], height=100, disabled=True)
                            
                            with col2:
                                st.metric("Similarity", f"{result['similarity_score']}%")
                                st.metric("Confidence", result['confidence'])
                                
                                if result['is_paraphrase']:
                                    st.success("✅ PARAPHRASE")
                                else:
                                    st.warning("❌ NOT PARAPHRASE")
                else:
                    st.warning("Please enter at least one comparison text")
            else:
                st.warning("Please enter both reference text and comparison texts")
    
    # Add information about the model
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About the Model")
    st.sidebar.markdown("""
    This app uses the **all-MiniLM-L6-v2** model which is:
    - Fast and efficient
    - Good at semantic understanding
    - Optimized for similarity tasks
    - Size: ~80MB
    - Context window: 256 tokens
    """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ using Streamlit and Sentence-Transformers")

if __name__ == "__main__":
    main()