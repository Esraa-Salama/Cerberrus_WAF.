"""
File: request_parser.py
Role: Request analysis and feature extraction for CerberusWAF.

This module handles the parsing and analysis of HTTP requests:
- Tokenizes and normalizes request data (headers, body, URL parameters)
- Extracts features for ML model input
- Detects known attack patterns using regex
- Builds vector representations of requests
- Performs initial security checks

Key Components:
- Request tokenization and normalization
- Feature extraction and vectorization
- Pattern matching for known attacks
- URL and parameter analysis
- Header and body content inspection

Dependencies:
- sklearn: For TF-IDF vectorization
- numpy: For numerical operations
- re: For pattern matching
"""

import re
from typing import Dict, List, Any, Tuple, Union
from urllib.parse import urlparse, parse_qs, unquote
import json
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class RequestParser:
    def __init__(self):
        """Initialize the request parser with common attack patterns."""
        self.attack_patterns = {
            'sql_injection': [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b.*\b(FROM|INTO|WHERE)\b)",
                r"(\b(OR|AND)\b.*\b(1=1|'1'='1')\b)",
                r"(\b(EXEC|EXECUTE)\b.*\b(SP|XP)\b)",
            ],
            'xss': [
                r"(<script.*?>.*?</script>)",
                r"(javascript:.*?)",
                r"(on\w+\s*=\s*['\"].*?['\"])",
            ],
            'path_traversal': [
                r"(\.\.\/\.\.\/)",
                r"(\.\.\\\.\.\\)",
                r"(\/etc\/passwd|\/etc\/shadow)",
            ],
            'command_injection': [
                r"(\b(cat|ls|rm|wget|curl|bash|sh)\b.*\b(>|<|\||;)\b)",
                r"(\b(system|exec|eval|shell_exec)\b.*\()",
            ]
        }

        # Initialize TF-IDF vectorizer for text analysis
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

        self.stop_words = set(stopwords.words('english'))

    def parse_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and analyze an HTTP request."""
        features = {
            'url_features': self._extract_url_features(request_data.get('path', '')),
            'header_features': self._extract_header_features(request_data.get('headers', {})),
            'body_features': self._extract_body_features(request_data.get('form', {}), request_data.get('json', {})),
            'query_features': self._extract_query_features(request_data.get('args', {})),
            'attack_indicators': self._check_attack_patterns(request_data)
        }

        return features

    def _extract_url_features(self, path: str) -> Dict[str, Any]:
        """Extract features from URL path."""
        parsed = urlparse(path)
        return {
            'path_length': len(path),
            'num_slashes': path.count('/'),
            'num_dots': path.count('.'),
            'num_parameters': len(parse_qs(parsed.query)),
            'file_extension': self._get_file_extension(path),
            'is_api_endpoint': self._is_api_endpoint(path)
        }

    def _extract_header_features(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Extract features from request headers."""
        features = {
            'content_type': headers.get('Content-Type', ''),
            'user_agent': headers.get('User-Agent', ''),
            'referer': headers.get('Referer', ''),
            'num_headers': len(headers),
            'has_suspicious_headers': self._check_suspicious_headers(headers)
        }
        return features

    def _extract_body_features(self, form_data: Dict[str, Any], json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from request body."""
        body_text = ''
        if form_data:
            body_text += ' '.join(str(v) for v in form_data.values())
        if json_data:
            body_text += ' '.join(str(v) for v in json_data.values())

        # Vectorize body text
        try:
            vector = self.vectorizer.fit_transform([body_text])
            features = vector.toarray()[0]
        except:
            features = np.zeros(1000)  # Default vector if text is empty

        return {
            'body_length': len(body_text),
            'num_fields': len(form_data) + len(json_data),
            'vector_features': features.tolist()
        }

    def _extract_query_features(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from query parameters."""
        features = {
            'num_params': len(query_params),
            'param_lengths': [len(str(v)) for v in query_params.values()],
            'has_suspicious_params': self._check_suspicious_params(query_params)
        }
        return features

    def _check_attack_patterns(self, request_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Check for known attack patterns in the request."""
        findings = defaultdict(list)

        # Combine all request data into a single string for pattern matching
        request_text = ' '.join([
            str(request_data.get('path', '')),
            str(request_data.get('args', {})),
            str(request_data.get('form', {})),
            str(request_data.get('json', {})),
            str(request_data.get('headers', {}))
        ])

        # Check each attack pattern category
        for attack_type, patterns in self.attack_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, request_text, re.IGNORECASE)
                for match in matches:
                    findings[attack_type].append(match.group(0))

        return dict(findings)

    def _get_file_extension(self, path: str) -> str:
        """Extract file extension from path."""
        match = re.search(r'\.([^./]+)$', path)
        return match.group(1) if match else ''

    def _is_api_endpoint(self, path: str) -> bool:
        """Check if the path is likely an API endpoint."""
        api_indicators = ['/api/', '/v1/', '/v2/', '/rest/']
        return any(indicator in path.lower() for indicator in api_indicators)

    def _check_suspicious_headers(self, headers: Dict[str, str]) -> bool:
        """Check for suspicious header patterns."""
        suspicious_patterns = [
            r'\.\.\/',
            r'<script',
            r'javascript:',
            r'eval\(',
            r'exec\('
        ]

        header_text = ' '.join(str(v) for v in headers.values())
        return any(re.search(pattern, header_text, re.IGNORECASE) for pattern in suspicious_patterns)

    def _check_suspicious_params(self, params: Dict[str, Any]) -> bool:
        """Check for suspicious parameter patterns."""
        suspicious_patterns = [
            r'\.\.\/',
            r'<script',
            r'javascript:',
            r'eval\(',
            r'exec\(',
            r'SELECT.*FROM',
            r'UNION.*SELECT'
        ]

        param_text = ' '.join(str(v) for v in params.values())
        return any(re.search(pattern, param_text, re.IGNORECASE) for pattern in suspicious_patterns)

    def vectorize_request(self, request_data: Dict[str, Any]) -> np.ndarray:
        """
        Convert HTTP request into a feature vector suitable for ML model input.

        Args:
            request_data: Dictionary containing request components

        Returns:
            numpy.ndarray: Combined feature vector
        """
        # Extract and normalize text from different components
        components = {
            'url': self._normalize_url(request_data.get('path', '')),
            'headers': self._normalize_headers(request_data.get('headers', {})),
            'query': self._normalize_query_params(request_data.get('args', {})),
            'body': self._normalize_body(request_data.get('form', {}), request_data.get('json', {}))
        }

        # Tokenize and vectorize each component
        vectors = {}
        for component, text in components.items():
            tokens = self._tokenize_text(text)
            vectors[component] = self._vectorize_tokens(tokens)

        # Combine vectors with appropriate weights
        weights = {
            'url': 0.3,
            'headers': 0.2,
            'query': 0.2,
            'body': 0.3
        }

        combined_vector = np.zeros(1000)  # Assuming 1000 features
        for component, vector in vectors.items():
            combined_vector += vector * weights[component]

        return combined_vector

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for analysis."""
        # Decode URL-encoded characters
        url = unquote(url)

        # Remove common URL patterns that don't contribute to analysis
        url = re.sub(r'https?://', '', url)
        url = re.sub(r'www\.', '', url)

        # Split path components
        path_components = url.split('/')

        # Remove empty components and common file extensions
        path_components = [
            re.sub(r'\.(html|php|asp|jsp|js|css|jpg|png|gif)$', '', comp)
            for comp in path_components if comp
        ]

        return ' '.join(path_components)

    def _normalize_headers(self, headers: Dict[str, str]) -> str:
        """Normalize HTTP headers for analysis."""
        # Convert headers to lowercase
        normalized = []
        for key, value in headers.items():
            # Skip common headers that don't contribute to analysis
            if key.lower() in {'content-length', 'connection', 'date', 'server'}:
                continue
            normalized.append(f"{key.lower()}:{value.lower()}")

        return ' '.join(normalized)

    def _normalize_query_params(self, params: Dict[str, Any]) -> str:
        """Normalize query parameters for analysis."""
        normalized = []
        for key, value in params.items():
            # Decode URL-encoded values
            key = unquote(str(key))
            value = unquote(str(value))

            # Remove common parameter patterns
            if re.match(r'^[a-z0-9_]+$', key):
                normalized.append(f"{key}:{value}")

        return ' '.join(normalized)

    def _normalize_body(self, form_data: Dict[str, Any], json_data: Dict[str, Any]) -> str:
        """Normalize request body for analysis."""
        normalized = []

        # Process form data
        for key, value in form_data.items():
            normalized.append(f"{key}:{value}")

        # Process JSON data
        if json_data:
            try:
                # Flatten nested JSON
                flattened = self._flatten_json(json_data)
                normalized.extend(f"{k}:{v}" for k, v in flattened.items())
            except:
                normalized.append(str(json_data))

        return ' '.join(normalized)

    def _flatten_json(self, data: Dict[str, Any], prefix: str = '') -> Dict[str, str]:
        """Flatten nested JSON structure."""
        flattened = {}
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_json(value, new_key))
            else:
                flattened[new_key] = str(value)
        return flattened

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words, removing stopwords and special characters."""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and short tokens
        tokens = [
            token for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return tokens

    def _vectorize_tokens(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to TF-IDF vector."""
        try:
            # Join tokens back into text for TF-IDF
            text = ' '.join(tokens)
            vector = self.vectorizer.fit_transform([text])
            return vector.toarray()[0]
        except:
            return np.zeros(1000)  # Return zero vector if vectorization fails
