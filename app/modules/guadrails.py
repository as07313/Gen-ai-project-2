from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory

class Guadrails:
    def __init__(self, api_key: str, endpoint: str):
        self.client = ContentSafetyClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
        self.severity_threshold = 4  # Default threshold
        
    def update_severity_threshold(self, threshold: int):
        """Update the severity threshold for content filtering"""
        self.severity_threshold = threshold
        print(f"Updated content safety threshold to {threshold}")

    def analyze_text(self, text: str):
        """Analyze text for harmful content"""
        try:
            # Fix: Handle Azure API changes properly
            request = AnalyzeTextOptions(text=text)
            response = self.client.analyze_text(request)
            
            # Return detailed analysis for logging and decision-making
            results = {}
            
            # Handle different API structures
            if hasattr(response, 'categories_analysis'):
                # Current API structure
                for category_result in response.categories_analysis:
                    # Check if category is a TextCategory enum or an object with name
                    if hasattr(category_result.category, 'name'):
                        category_name = category_result.category.name.lower()
                    # Check if it's a string
                    elif isinstance(category_result.category, str):
                        category_name = category_result.category.lower()
                    # Fallback for any other case
                    else:
                        category_name = str(category_result.category).lower()
                        
                    results[category_name] = category_result.severity
                    if category_result.severity > self.severity_threshold:
                        return False, results
            else:
                # Fallback for other API structures
                for category in ['hate', 'self_harm', 'sexual', 'violence']:
                    if hasattr(response, category):
                        category_obj = getattr(response, category)
                        if category_obj and hasattr(category_obj, 'severity'):
                            results[category] = category_obj.severity
                            if category_obj.severity > self.severity_threshold:
                                return False, results
            
            return True, results
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return False, {"error": str(e)}
    def check_input_query(self, query: str):
        """Validate user query safety"""
        is_safe, results = self.analyze_text(query)
        if not is_safe:
            return False, "I cannot process queries with potentially harmful content. Please rephrase your question."
        return True, query

    def check_retrieved_documents(self, documents: list):
        """Filter retrieved documents for safety and relevance"""
        safe_documents = []
        for doc in documents:
            is_safe, _ = self.analyze_text(doc.page_content)
            if is_safe:
                safe_documents.append(doc)
        
        return safe_documents

    def check_output_safety(self, generated_response):
        """Check if the AI-generated response contains harmful content"""
        is_safe, results = self.analyze_text(generated_response)
        if not is_safe:
            return False, "I apologize, but I need to refine my response to ensure it's safe and accurate for maternal health information."
        return True, results