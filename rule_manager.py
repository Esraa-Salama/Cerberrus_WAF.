"""
File: rule_manager.py
Role: Rule management interface for CerberusWAF.

This module implements a web interface for managing WAF rules:
- Add new rules
- Edit existing rules
- Remove rules
- Enable/disable rules
- Rule priority management

Key Components:
- Flask web application
- Rule CRUD operations
- Rule validation
- Rule persistence

Dependencies:
- flask: Web framework
- json: Data serialization
- datetime: Time handling
"""

from flask import Flask, render_template, jsonify, request
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import re

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Rule storage
RULES_FILE = "rules.json"


class Rule:
    """WAF Rule class for managing individual rules."""

    def __init__(self,
                 name: str,
                 pattern: str,
                 action: str,
                 priority: int,
                 description: str = "",
                 enabled: bool = True):
        """
        Initialize a WAF rule.

        Args:
            name: Unique rule name
            pattern: Regex pattern to match
            action: Action to take (allow/block/alert)
            priority: Rule priority (higher = more important)
            description: Rule description
            enabled: Whether rule is enabled
        """
        self.name = name
        self.pattern = pattern
        self.action = action.lower()
        self.priority = priority
        self.description = description
        self.enabled = enabled
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            'name': self.name,
            'pattern': self.pattern,
            'action': self.action,
            'priority': self.priority,
            'description': self.description,
            'enabled': self.enabled,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Rule':
        """Create rule from dictionary."""
        rule = cls(
            name=data['name'],
            pattern=data['pattern'],
            action=data['action'],
            priority=data['priority'],
            description=data.get('description', ''),
            enabled=data.get('enabled', True)
        )
        rule.created_at = data.get('created_at', datetime.now().isoformat())
        rule.updated_at = data.get('updated_at', rule.created_at)
        return rule

    def validate(self) -> List[str]:
        """Validate rule configuration."""
        errors = []

        # Validate name
        if not self.name or not re.match(r'^[a-zA-Z0-9_-]+$', self.name):
            errors.append(
                "Rule name must be alphanumeric with underscores and hyphens")

        # Validate pattern
        try:
            re.compile(self.pattern)
        except re.error:
            errors.append("Invalid regex pattern")

        # Validate action
        if self.action not in ['allow', 'block', 'alert']:
            errors.append("Action must be one of: allow, block, alert")

        # Validate priority
        if not isinstance(self.priority, int) or self.priority < 0:
            errors.append("Priority must be a non-negative integer")

        return errors


class RuleManager:
    """Manager class for WAF rules."""

    def __init__(self, rules_file: str = RULES_FILE):
        """Initialize rule manager."""
        self.rules_file = rules_file
        self.rules: Dict[str, Rule] = {}
        self.load_rules()

    def load_rules(self):
        """Load rules from file."""
        try:
            if Path(self.rules_file).exists():
                with open(self.rules_file, 'r') as f:
                    data = json.load(f)
                    self.rules = {
                        name: Rule.from_dict(rule_data)
                        for name, rule_data in data.items()
                    }
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            self.rules = {}

    def save_rules(self):
        """Save rules to file."""
        try:
            data = {
                name: rule.to_dict()
                for name, rule in self.rules.items()
            }
            with open(self.rules_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save rules: {e}")

    def add_rule(self, rule: Rule) -> List[str]:
        """Add a new rule."""
        errors = rule.validate()
        if errors:
            return errors

        if rule.name in self.rules:
            return ["Rule with this name already exists"]

        self.rules[rule.name] = rule
        self.save_rules()
        return []

    def update_rule(self, name: str, updates: Dict[str, Any]) -> List[str]:
        """Update an existing rule."""
        if name not in self.rules:
            return ["Rule not found"]

        rule = self.rules[name]

        # Create updated rule
        updated_rule = Rule(
            name=updates.get('name', rule.name),
            pattern=updates.get('pattern', rule.pattern),
            action=updates.get('action', rule.action),
            priority=updates.get('priority', rule.priority),
            description=updates.get('description', rule.description),
            enabled=updates.get('enabled', rule.enabled)
        )

        errors = updated_rule.validate()
        if errors:
            return errors

        # Update rule
        self.rules[name] = updated_rule
        updated_rule.updated_at = datetime.now().isoformat()
        self.save_rules()
        return []

    def delete_rule(self, name: str) -> bool:
        """Delete a rule."""
        if name in self.rules:
            del self.rules[name]
            self.save_rules()
            return True
        return False

    def get_rule(self, name: str) -> Optional[Rule]:
        """Get a rule by name."""
        return self.rules.get(name)

    def get_all_rules(self) -> List[Rule]:
        """Get all rules."""
        return list(self.rules.values())

    def enable_rule(self, name: str) -> bool:
        """Enable a rule."""
        if name in self.rules:
            self.rules[name].enabled = True
            self.rules[name].updated_at = datetime.now().isoformat()
            self.save_rules()
            return True
        return False

    def disable_rule(self, name: str) -> bool:
        """Disable a rule."""
        if name in self.rules:
            self.rules[name].enabled = False
            self.rules[name].updated_at = datetime.now().isoformat()
            self.save_rules()
            return True
        return False


# Initialize rule manager
rule_manager = RuleManager()


@app.route('/rules')
def index():
    """Render the rule management page."""
    return render_template('rule_manager.html')


@app.route('/api/rules', methods=['GET'])
def get_rules():
    """Get all rules."""
    rules = [rule.to_dict() for rule in rule_manager.get_all_rules()]
    return jsonify(rules)


@app.route('/api/rules', methods=['POST'])
def create_rule():
    """Create a new rule."""
    data = request.json
    rule = Rule(
        name=data['name'],
        pattern=data['pattern'],
        action=data['action'],
        priority=data['priority'],
        description=data.get('description', ''),
        enabled=data.get('enabled', True)
    )

    errors = rule_manager.add_rule(rule)
    if errors:
        return jsonify({'errors': errors}), 400

    return jsonify(rule.to_dict()), 201


@app.route('/api/rules/<name>', methods=['PUT'])
def update_rule(name):
    """Update an existing rule."""
    data = request.json
    errors = rule_manager.update_rule(name, data)

    if errors:
        return jsonify({'errors': errors}), 400

    return jsonify(rule_manager.get_rule(name).to_dict())


@app.route('/api/rules/<name>', methods=['DELETE'])
def delete_rule(name):
    """Delete a rule."""
    if rule_manager.delete_rule(name):
        return '', 204
    return jsonify({'error': 'Rule not found'}), 404


@app.route('/api/rules/<name>/enable', methods=['POST'])
def enable_rule(name):
    """Enable a rule."""
    if rule_manager.enable_rule(name):
        return jsonify({'status': 'enabled'})
    return jsonify({'error': 'Rule not found'}), 404


@app.route('/api/rules/<name>/disable', methods=['POST'])
def disable_rule(name):
    """Disable a rule."""
    if rule_manager.disable_rule(name):
        return jsonify({'status': 'disabled'})
    return jsonify({'error': 'Rule not found'}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5001)
