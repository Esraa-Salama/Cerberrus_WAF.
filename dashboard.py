"""
File: dashboard.py
Role: Flask dashboard for CerberusWAF monitoring and statistics.

This module implements a web dashboard for monitoring WAF activity:
- Real-time traffic statistics
- Attack type distribution
- Recent request logs
- System status monitoring

Key Components:
- Flask web application
- Real-time data endpoints
- Data aggregation utilities
- Template rendering

Dependencies:
- flask: Web framework
- flask_socketio: WebSocket support
- json: Data serialization
- datetime: Time handling
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from collections import defaultdict

# Import WAF components
from traffic_monitor import TrafficMonitor
from evaluate_core import run_evaluation

app = Flask(__name__)
socketio = SocketIO(app)
logger = logging.getLogger(__name__)

# Initialize WAF components
traffic_monitor = TrafficMonitor()


@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/stats/global')
def get_global_stats():
    """Get global traffic statistics."""
    stats = traffic_monitor.get_global_stats()
    return jsonify(stats)


@app.route('/api/stats/attack_types')
def get_attack_types():
    """Get distribution of attack types."""
    alerts = traffic_monitor.get_recent_alerts(minutes=60)
    attack_types = defaultdict(int)

    for alert in alerts:
        if alert['level'] in ['warning', 'critical']:
            attack_types[alert['type']] += 1

    return jsonify({
        'labels': list(attack_types.keys()),
        'data': list(attack_types.values())
    })


@app.route('/api/requests/recent')
def get_recent_requests():
    """Get recent request logs with classification results."""
    # Get recent requests from traffic monitor
    recent_requests = []
    for ip in traffic_monitor.request_history:
        stats = traffic_monitor.get_client_stats(ip)
        if stats['last_request']:
            recent_requests.append({
                'ip': ip,
                'timestamp': stats['last_request'],
                'requests_per_second': stats['requests_per_second'],
                'is_blacklisted': stats['is_blacklisted'],
                'violations': stats['violations']
            })

    # Sort by timestamp
    recent_requests.sort(key=lambda x: x['timestamp'], reverse=True)

    return jsonify(recent_requests[:50])  # Return last 50 requests


@app.route('/api/alerts/recent')
def get_recent_alerts():
    """Get recent security alerts."""
    alerts = traffic_monitor.get_recent_alerts(minutes=5)
    return jsonify(alerts)


@app.route('/evaluate')
def evaluation():
    result = run_evaluation()
    return jsonify(result)


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    logger.info("Client connected")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    logger.info("Client disconnected")


def emit_stats_update():
    """Emit updated statistics to connected clients."""
    stats = traffic_monitor.get_global_stats()
    socketio.emit('stats_update', stats)


def emit_alert(alert: Dict[str, Any]):
    """Emit new alert to connected clients."""
    socketio.emit('new_alert', alert)


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
