"""
File: traffic_monitor.py
Role: Traffic analysis and anomaly detection for CerberusWAF.

This module monitors and analyzes web traffic patterns:
- Tracks request rates and patterns per IP
- Detects anomalous traffic spikes
- Monitors endpoint usage and response times
- Generates alerts for suspicious activity
- Maintains traffic statistics and history

Key Components:
- Request rate monitoring
- Anomaly detection
- Alert generation
- Traffic statistics
- Historical data management

Dependencies:
- collections: For data structures
- datetime: For timestamp handling
- logging: For alert logging
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from threading import Lock
import statistics

logger = logging.getLogger(__name__)


class TrafficMonitor:
    def __init__(self, window_size: int = 3600):
        """
        Initialize the traffic monitor.

        Args:
            window_size: Time window in seconds for traffic analysis (default: 1 hour)
        """
        self.window_size = window_size
        self.request_history = defaultdict(
            lambda: deque(maxlen=1000))  # IP -> list of timestamps
        self.endpoint_stats = defaultdict(lambda: {
            'count': 0,
            'errors': 0,
            'avg_response_time': 0,
            'last_request': None,
            # Store recent request timestamps
            'request_times': deque(maxlen=1000)
        })
        self.anomaly_thresholds = {
            'requests_per_minute': 100,
            'requests_per_second': 10,
            'error_rate': 0.1,  # 10% error rate threshold
            'response_time': 1000,  # 1 second threshold
            'unique_endpoints': 50,  # Maximum unique endpoints per minute
            'burst_threshold': 20,  # Requests in 1 second
            'sustained_threshold': 1000,  # Requests in 1 minute
            'ip_blacklist_threshold': 3  # Number of violations before blacklisting
        }
        self.blacklisted_ips = set()
        self.ip_violations = defaultdict(int)
        self.ip_violation_history = defaultdict(list)
        self.global_stats = {
            'total_requests': 0,
            'total_errors': 0,
            'peak_requests_per_second': 0,
            'current_requests_per_second': 0,
            # Last hour of request timestamps
            'request_times': deque(maxlen=3600)
        }
        self.lock = Lock()
        self.alerts = []
        self.alert_levels = {
            'info': 0,
            'warning': 1,
            'critical': 2
        }

    def record_request(self, request_data: Dict[str, Any], response_time: float):
        """Record a new request and its metrics."""
        with self.lock:
            client_ip = request_data.get('ip', 'unknown')
            endpoint = request_data.get('path', 'unknown')
            timestamp = time.time()

            # Skip if IP is blacklisted
            if client_ip in self.blacklisted_ips:
                return

            # Update global statistics
            self._update_global_stats(timestamp)

            # Record request timestamp
            self.request_history[client_ip].append(timestamp)

            # Update endpoint statistics
            self._update_endpoint_stats(endpoint, response_time, timestamp)

            # Check for anomalies
            self._check_anomalies(client_ip, endpoint,
                                  response_time, timestamp)

    def _update_global_stats(self, timestamp: float):
        """Update global traffic statistics."""
        self.global_stats['total_requests'] += 1
        self.global_stats['request_times'].append(timestamp)

        # Calculate current requests per second
        current_time = time.time()
        recent_requests = [t for t in self.global_stats['request_times']
                           if current_time - t < 1]
        self.global_stats['current_requests_per_second'] = len(recent_requests)

        # Update peak if necessary
        if len(recent_requests) > self.global_stats['peak_requests_per_second']:
            self.global_stats['peak_requests_per_second'] = len(
                recent_requests)

    def _update_endpoint_stats(self, endpoint: str, response_time: float, timestamp: float):
        """Update statistics for an endpoint."""
        stats = self.endpoint_stats[endpoint]
        stats['count'] += 1
        stats['last_request'] = datetime.fromtimestamp(timestamp)
        stats['request_times'].append(timestamp)

        # Update average response time using exponential moving average
        alpha = 0.1  # Smoothing factor
        stats['avg_response_time'] = (
            alpha * response_time +
            (1 - alpha) * stats['avg_response_time']
        )

    def _check_anomalies(self, client_ip: str, endpoint: str, response_time: float, timestamp: float):
        """Check for various types of anomalies."""
        current_time = time.time()

        # Get recent requests for this IP
        recent_requests = [
            t for t in self.request_history[client_ip]
            if current_time - t < 60
        ]

        # Check for burst attacks (high rate in short time)
        burst_requests = [t for t in recent_requests if current_time - t < 1]
        if len(burst_requests) > self.anomaly_thresholds['burst_threshold']:
            self._handle_violation(client_ip, 'burst_attack',
                                   f"Burst attack detected: {len(burst_requests)} requests/second")

        # Check for sustained attacks (high rate over longer period)
        if len(recent_requests) > self.anomaly_thresholds['sustained_threshold']:
            self._handle_violation(client_ip, 'sustained_attack',
                                   f"Sustained attack detected: {len(recent_requests)} requests/minute")

        # Check response time anomalies
        if response_time > self.anomaly_thresholds['response_time']:
            self._create_alert('high_response_time',
                               f"Endpoint {endpoint} response time: {response_time}ms",
                               level='warning')

        # Check for endpoint scanning
        unique_endpoints = len(set(
            endpoint for endpoint in self.endpoint_stats.keys()
            if self.endpoint_stats[endpoint]['last_request'] and
            (datetime.now() -
             self.endpoint_stats[endpoint]['last_request']) < timedelta(minutes=1)
        ))
        if unique_endpoints > self.anomaly_thresholds['unique_endpoints']:
            self._handle_violation(client_ip, 'endpoint_scanning',
                                   f"Endpoint scanning detected: {unique_endpoints} unique endpoints/minute")

    def _handle_violation(self, client_ip: str, violation_type: str, message: str):
        """Handle security violations and update violation history."""
        self.ip_violations[client_ip] += 1
        self.ip_violation_history[client_ip].append({
            'timestamp': datetime.now().isoformat(),
            'type': violation_type,
            'message': message
        })

        # Create alert
        self._create_alert(violation_type, message, level='warning')

        # Check if IP should be blacklisted
        if self.ip_violations[client_ip] >= self.anomaly_thresholds['ip_blacklist_threshold']:
            self.blacklisted_ips.add(client_ip)
            self._create_alert('ip_blacklisted',
                               f"IP {client_ip} has been blacklisted due to multiple violations",
                               level='critical')

    def _create_alert(self, alert_type: str, message: str, level: str = 'info'):
        """Create and log a new alert with severity level."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'level': level
        }
        self.alerts.append(alert)
        logger.warning(f"Traffic Monitor Alert: {json.dumps(alert)}")

    def get_client_stats(self, client_ip: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific client IP."""
        with self.lock:
            recent_requests = [
                t for t in self.request_history[client_ip]
                if time.time() - t < self.window_size
            ]

            # Calculate request rates
            current_time = time.time()
            requests_per_second = len(
                [t for t in recent_requests if current_time - t < 1])
            requests_per_minute = len(
                [t for t in recent_requests if current_time - t < 60])

            return {
                'total_requests': len(recent_requests),
                'requests_per_second': requests_per_second,
                'requests_per_minute': requests_per_minute,
                'requests_per_hour': len([t for t in recent_requests if current_time - t < 3600]),
                'last_request': datetime.fromtimestamp(recent_requests[-1]).isoformat() if recent_requests else None,
                'violations': self.ip_violations[client_ip],
                'is_blacklisted': client_ip in self.blacklisted_ips,
                'violation_history': self.ip_violation_history[client_ip]
            }

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global traffic statistics."""
        with self.lock:
            return {
                'total_requests': self.global_stats['total_requests'],
                'current_requests_per_second': self.global_stats['current_requests_per_second'],
                'peak_requests_per_second': self.global_stats['peak_requests_per_second'],
                'total_errors': self.global_stats['total_errors'],
                'blacklisted_ips': len(self.blacklisted_ips),
                'active_endpoints': len(self.endpoint_stats)
            }

    def get_endpoint_stats(self, endpoint: str) -> Dict[str, Any]:
        """Get statistics for a specific endpoint."""
        return self.endpoint_stats[endpoint]

    def get_recent_alerts(self, minutes: int = 5, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts from the last N minutes, optionally filtered by level."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff
        ]

        if level:
            alerts = [alert for alert in alerts if alert['level'] == level]

        return alerts

    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update anomaly detection thresholds."""
        with self.lock:
            self.anomaly_thresholds.update(new_thresholds)
            logger.info(
                f"Updated anomaly thresholds: {json.dumps(new_thresholds)}")

    def clear_history(self):
        """Clear all request history and statistics."""
        with self.lock:
            self.request_history.clear()
            self.endpoint_stats.clear()
            self.alerts.clear()
            self.ip_violations.clear()
            self.ip_violation_history.clear()
            self.blacklisted_ips.clear()
            self.global_stats['total_requests'] = 0
            self.global_stats['request_times'].clear()
            logger.info("Cleared all traffic monitoring history")

    def remove_from_blacklist(self, client_ip: str):
        """Remove an IP from the blacklist and reset its violations."""
        with self.lock:
            if client_ip in self.blacklisted_ips:
                self.blacklisted_ips.remove(client_ip)
                self.ip_violations[client_ip] = 0
                self._create_alert('ip_unblacklisted',
                                   f"IP {client_ip} has been removed from blacklist",
                                   level='info')
