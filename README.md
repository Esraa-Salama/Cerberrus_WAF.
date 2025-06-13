"""
CerberusWAF: AI-powered Web Application Firewall

Modules:

- waf_engine.py: intercepts HTTP requests, integrates with AI prediction.
- request_parser.py: tokenizes request data and builds vector input.
- traffic_monitor.py: logs and detects anomalous traffic spikes.
- webapp/: GUI for monitoring, rule editing, system stats.
  Uses Flask, PyTorch, Transformers.
  """

# CerberusWAF - AI-Enhanced Web Application Firewall

CerberusWAF is a next-generation Web Application Firewall that combines traditional rule-based security with advanced machine learning to protect web applications from both known and zero-day attacks.

## Features

- Real-time HTTP(S) traffic analysis
- Deep learning-based attack detection (CNN, LSTM, Transformer models)
- Traditional rule-based security policies
- Real-time traffic monitoring and anomaly detection
- Web-based dashboard for monitoring and management
- Scalable deployment with Docker and Kubernetes
- SSL/TLS certificate management
- Comprehensive audit logging

## Architecture

The system consists of several key components:

- **WAF Engine**: Core middleware for request interception and processing
- **Request Parser**: HTTP request analysis and feature extraction
- **Traffic Monitor**: Real-time traffic analysis and anomaly detection
- **ML Models**: Deep learning models for attack classification
- **Web Interface**: Flask-based dashboard for monitoring and management

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/CerberusWAF.git
cd CerberusWAF
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:

```bash
flask db upgrade
```

## Usage

1. Start the WAF:

```bash
python run.py
```

2. Access the web interface at `http://localhost:5000`

3. Configure your web server to proxy requests through the WAF

## Development

### Project Structure

```
CerberusWAF/
├── waf_engine.py          # Core WAF middleware
├── request_parser.py      # Request analysis
├── traffic_monitor.py     # Traffic monitoring
├── models/               # ML model implementations
├── webapp/              # Flask web interface
├── deployment/          # Docker and K8s configs
└── tests/              # Test suite
```

### Running Tests

```bash
pytest tests/
```

## Security

- All sensitive data is encrypted at rest
- Regular security audits and penetration testing
- Automatic updates for ML models
- Comprehensive logging and monitoring

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
