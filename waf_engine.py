"""
File: waf_engine.py
Role: Main traffic interceptor for CerberusWAF.
Routes requests through parser and prediction engine.

This module implements the core Web Application Firewall functionality:
- Intercepts and analyzes all incoming HTTP(S) requests
- Applies rate limiting and IP blocking
- Routes requests through the parser for feature extraction
- Integrates with ML models for attack detection
- Enforces security policies based on analysis results

Key Components:
- Request interception and preprocessing
- Rate limiting and IP management
- Request logging and monitoring
- Security policy enforcement
- Integration with ML models (to be implemented)

Dependencies:
- Flask: Web framework for request handling
- RequestParser: For request analysis and feature extraction
- TrafficMonitor: For traffic analysis and anomaly detection
"""

import logging
from typing import Dict, Any, Optional, Tuple
from flask import Flask, request, Response, redirect
import json
import time
from datetime import datetime
import torch
import numpy as np
from request_parser import RequestParser
from traffic_monitor import TrafficMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CerberusWAF:
    def __init__(self, app: Optional[Flask] = None, model_path: str = None):
        """Initialize the WAF engine with optional ML model."""
        self.app = app
        self.blocked_ips = set()
        self.request_history = {}
        self.rate_limits = {
            'requests_per_minute': 100,
            'requests_per_hour': 1000
        }

        # Initialize components
        self.parser = RequestParser()
        self.monitor = TrafficMonitor()
        self.model = self._load_model(model_path) if model_path else None

        # WAF configuration
        self.config = {
            'block_threshold': 0.8,  # Confidence threshold for blocking
            'redirect_threshold': 0.6,  # Confidence threshold for redirecting
            'redirect_url': '/waf/challenge',  # URL for challenge page
            'enable_ai': True,  # Whether to use AI model
            'enable_rules': True,  # Whether to use rule-based checks
        }

        if app is not None:
            self.init_app(app)

    def _load_model(self, model_path: str) -> Optional[torch.nn.Module]:
        """Load the pre-trained ML model."""
        try:
            model = torch.load(model_path)
            model.eval()  # Set to evaluation mode
            logger.info(f"Loaded ML model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return None

    def init_app(self, app: Flask):
        """Initialize the WAF with a Flask application."""
        self.app = app

        # Register middleware
        @app.before_request
        def before_request():
            return self.process_request()

        # Register error handlers
        @app.errorhandler(403)
        def forbidden_error(error):
            return self.handle_blocked_request()

    def process_request(self) -> Optional[Response]:
        """Process incoming requests and apply security rules."""
        client_ip = request.remote_addr
        request_id = f"{client_ip}-{int(time.time())}"

        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            logger.warning(f"Blocked request from blacklisted IP: {client_ip}")
            return self.handle_blocked_request()

        # Rate limiting
        if not self._check_rate_limits(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return self.handle_rate_limit_exceeded()

        # Extract request data
        request_data = self._extract_request_data(request_id, client_ip)

        # Parse and analyze request
        features = self.parser.parse_request(request_data)

        # Get prediction from ML model
        prediction = self._get_model_prediction(
            features) if self.config['enable_ai'] else None

        # Apply WAF logic
        action = self._determine_action(features, prediction)

        # Log request and update monitoring
        self._log_request(request_data)
        self.monitor.record_request(request_data, time.time())

        return self._execute_action(action)

    def _extract_request_data(self, request_id: str, client_ip: str) -> Dict[str, Any]:
        """Extract and structure request data."""
        return {
            'id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'ip': client_ip,
            'method': request.method,
            'path': request.path,
            'headers': dict(request.headers),
            'args': dict(request.args),
            'form': dict(request.form),
            'json': request.get_json(silent=True)
        }

    def _get_model_prediction(self, features: Dict[str, Any]) -> Tuple[float, str]:
        """Get prediction from ML model."""
        if not self.model:
            return 0.0, 'unknown'

        try:
            # Convert features to model input format
            model_input = self._prepare_model_input(features)

            # Get prediction
            with torch.no_grad():
                output = self.model(model_input)
                confidence = torch.sigmoid(output).item()
                prediction = 'malicious' if confidence > 0.5 else 'benign'

            return confidence, prediction
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return 0.0, 'unknown'

    def _prepare_model_input(self, features: Dict[str, Any]) -> torch.Tensor:
        """Convert features to model input format."""
        # Combine all feature vectors
        feature_vector = np.concatenate([
            features['url_features'].values(),
            features['header_features'].values(),
            features['body_features']['vector_features'],
            features['query_features'].values()
        ])

        return torch.FloatTensor(feature_vector).unsqueeze(0)

    def _determine_action(self, features: Dict[str, Any], prediction: Optional[Tuple[float, str]]) -> str:
        """Determine WAF action based on analysis results."""
        # Check rule-based patterns first
        if self.config['enable_rules']:
            if any(features['attack_indicators'].values()):
                return 'block'

        # Check ML model prediction
        if prediction and self.config['enable_ai']:
            confidence, label = prediction
            if label == 'malicious':
                if confidence >= self.config['block_threshold']:
                    return 'block'
                elif confidence >= self.config['redirect_threshold']:
                    return 'redirect'

        return 'allow'

    def _execute_action(self, action: str) -> Optional[Response]:
        """Execute the determined WAF action."""
        if action == 'block':
            return self.handle_blocked_request()
        elif action == 'redirect':
            return redirect(self.config['redirect_url'])
        return None

    def _check_rate_limits(self, client_ip: str) -> bool:
        """Check if the client has exceeded rate limits."""
        current_time = time.time()

        # Initialize client history if not exists
        if client_ip not in self.request_history:
            self.request_history[client_ip] = []

        # Clean old requests
        self.request_history[client_ip] = [
            t for t in self.request_history[client_ip]
            if current_time - t < 3600  # Keep last hour
        ]

        # Add current request
        self.request_history[client_ip].append(current_time)

        # Check limits
        requests_last_minute = len([t for t in self.request_history[client_ip]
                                    if current_time - t < 60])
        requests_last_hour = len(self.request_history[client_ip])

        return (requests_last_minute <= self.rate_limits['requests_per_minute'] and
                requests_last_hour <= self.rate_limits['requests_per_hour'])

    def _log_request(self, request_data: Dict[str, Any]):
        """Log request details for analysis."""
        logger.info(f"Request processed: {json.dumps(request_data)}")

    def handle_blocked_request(self) -> Response:
        """Handle blocked requests."""
        return Response(
            json.dumps({
                'error': 'Access denied',
                'message': 'Your request has been blocked by the WAF'
            }),
            status=403,
            mimetype='application/json'
        )

    def handle_rate_limit_exceeded(self) -> Response:
        """Handle rate limit exceeded requests."""
        return Response(
            json.dumps({
                'error': 'Rate limit exceeded',
                'message': 'Too many requests from your IP'
            }),
            status=429,
            mimetype='application/json'
        )

    def block_ip(self, ip: str, duration: int = 3600):
        """Block an IP address for a specified duration (in seconds)."""
        self.blocked_ips.add(ip)
        logger.info(f"IP {ip} blocked for {duration} seconds")

        # Schedule unblock
        def unblock():
            time.sleep(duration)
            self.blocked_ips.remove(ip)
            logger.info(f"IP {ip} unblocked")

        # TODO: Implement proper scheduling mechanism

    def update_rate_limits(self, requests_per_minute: int, requests_per_hour: int):
        """Update rate limiting thresholds."""
        self.rate_limits = {
            'requests_per_minute': requests_per_minute,
            'requests_per_hour': requests_per_hour
        }
        logger.info(f"Rate limits updated: {self.rate_limits}")

    def update_config(self, new_config: Dict[str, Any]):
        """Update WAF configuration."""
        self.config.update(new_config)
        logger.info(f"Updated WAF configuration: {json.dumps(new_config)}")
