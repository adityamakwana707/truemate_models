"""
TruthMate Monitoring Service
Real-time monitoring, metrics, and alerting for ML services
"""
from flask import Flask, jsonify, request, render_template_string
import psutil
import time
import json
import os
import requests
import threading
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
from typing import Dict, List
import sqlite3
import plotly.graph_objs as go
import plotly.utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class TruthMateMonitor:
    def __init__(self):
        self.metrics = defaultdict(deque)
        self.alerts = []
        self.services = {}
        self.db_path = 'monitoring/metrics.db'
        self.init_database()
        
        # Start background monitoring
        self.monitoring_thread = threading.Thread(target=self.collect_metrics, daemon=True)
        self.monitoring_thread.start()
        
    def init_database(self):
        """Initialize SQLite database for metrics storage"""
        os.makedirs('monitoring', exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                timestamp INTEGER,
                cpu_percent REAL,
                memory_percent REAL,
                disk_percent REAL,
                network_sent REAL,
                network_recv REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS service_metrics (
                timestamp INTEGER,
                service_name TEXT,
                response_time REAL,
                status_code INTEGER,
                error_message TEXT,
                requests_per_minute REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_metrics (
                timestamp INTEGER,
                service_name TEXT,
                model_name TEXT,
                inference_time REAL,
                confidence_score REAL,
                verdict TEXT,
                input_length INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                timestamp INTEGER,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                resolved INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def register_service(self, service_name: str, url: str, health_endpoint: str = '/health'):
        """Register a service for monitoring"""
        self.services[service_name] = {
            'url': url,
            'health_endpoint': health_endpoint,
            'last_check': 0,
            'status': 'unknown',
            'response_times': deque(maxlen=100),
            'error_count': 0,
            'request_count': 0
        }
        logger.info(f"Registered service: {service_name} at {url}")
    
    def collect_metrics(self):
        """Background thread for collecting metrics"""
        while True:
            try:
                # Collect system metrics
                self.collect_system_metrics()
                
                # Check service health
                for service_name in self.services:
                    self.check_service_health(service_name)
                
                # Store metrics to database
                self.store_metrics()
                
                # Check for alerts
                self.check_alerts()
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
            
            time.sleep(30)  # Collect every 30 seconds
    
    def collect_system_metrics(self):
        """Collect system-level metrics"""
        timestamp = int(time.time())
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network
        network = psutil.net_io_counters()
        
        metrics = {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent,
            'network_sent': network.bytes_sent,
            'network_recv': network.bytes_recv
        }
        
        self.metrics['system'].append(metrics)
        
        # Keep only last 1000 entries in memory
        if len(self.metrics['system']) > 1000:
            self.metrics['system'].popleft()
    
    def check_service_health(self, service_name: str):
        """Check health of a specific service"""
        service = self.services[service_name]
        
        try:
            start_time = time.time()
            url = f"{service['url']}{service['health_endpoint']}"
            
            response = requests.get(url, timeout=10)
            response_time = time.time() - start_time
            
            # Update service metrics
            service['last_check'] = int(time.time())
            service['status'] = 'healthy' if response.status_code == 200 else 'unhealthy'
            service['response_times'].append(response_time)
            service['request_count'] += 1
            
            # Store service metrics
            metrics = {
                'timestamp': int(time.time()),
                'service_name': service_name,
                'response_time': response_time,
                'status_code': response.status_code,
                'error_message': None
            }
            
            self.metrics[f'service_{service_name}'].append(metrics)
            
        except Exception as e:
            # Handle service errors
            service['status'] = 'error'
            service['error_count'] += 1
            service['last_check'] = int(time.time())
            
            metrics = {
                'timestamp': int(time.time()),
                'service_name': service_name,
                'response_time': 0,
                'status_code': 0,
                'error_message': str(e)
            }
            
            self.metrics[f'service_{service_name}'].append(metrics)
            
            logger.warning(f"Service {service_name} health check failed: {e}")
    
    def store_metrics(self):
        """Store metrics to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Store system metrics
            if 'system' in self.metrics and self.metrics['system']:
                latest_system = self.metrics['system'][-1]
                cursor.execute('''
                    INSERT INTO system_metrics 
                    (timestamp, cpu_percent, memory_percent, disk_percent, network_sent, network_recv)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    latest_system['timestamp'],
                    latest_system['cpu_percent'],
                    latest_system['memory_percent'], 
                    latest_system['disk_percent'],
                    latest_system['network_sent'],
                    latest_system['network_recv']
                ))
            
            # Store service metrics
            for key, metrics_deque in self.metrics.items():
                if key.startswith('service_') and metrics_deque:
                    latest_metric = metrics_deque[-1]
                    cursor.execute('''
                        INSERT INTO service_metrics
                        (timestamp, service_name, response_time, status_code, error_message)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        latest_metric['timestamp'],
                        latest_metric['service_name'],
                        latest_metric['response_time'],
                        latest_metric['status_code'],
                        latest_metric['error_message']
                    ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
        finally:
            conn.close()
    
    def check_alerts(self):
        """Check for alert conditions"""
        current_time = int(time.time())
        
        # System alerts
        if 'system' in self.metrics and self.metrics['system']:
            latest_system = self.metrics['system'][-1]
            
            # High CPU alert
            if latest_system['cpu_percent'] > 80:
                self.create_alert('high_cpu', 'warning', 
                                f"High CPU usage: {latest_system['cpu_percent']:.1f}%")
            
            # High memory alert  
            if latest_system['memory_percent'] > 85:
                self.create_alert('high_memory', 'warning',
                                f"High memory usage: {latest_system['memory_percent']:.1f}%")
            
            # High disk usage alert
            if latest_system['disk_percent'] > 90:
                self.create_alert('high_disk', 'critical',
                                f"High disk usage: {latest_system['disk_percent']:.1f}%")
        
        # Service alerts
        for service_name, service in self.services.items():
            # Service down alert
            if service['status'] == 'error' or service['status'] == 'unhealthy':
                self.create_alert('service_down', 'critical',
                                f"Service {service_name} is {service['status']}")
            
            # High response time alert
            if service['response_times'] and len(service['response_times']) > 0:
                avg_response_time = sum(service['response_times']) / len(service['response_times'])
                if avg_response_time > 10:  # 10 seconds
                    self.create_alert('slow_response', 'warning',
                                    f"Service {service_name} slow response: {avg_response_time:.2f}s")
    
    def create_alert(self, alert_type: str, severity: str, message: str):
        """Create a new alert"""
        # Check if similar alert already exists (avoid spam)
        recent_alerts = [a for a in self.alerts if 
                        a['alert_type'] == alert_type and 
                        time.time() - a['timestamp'] < 300]  # 5 minutes
        
        if recent_alerts:
            return  # Don't create duplicate alerts
        
        alert = {
            'timestamp': time.time(),
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'resolved': False
        }
        
        self.alerts.append(alert)
        
        # Store alert in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (timestamp, alert_type, severity, message)
            VALUES (?, ?, ?, ?)
        ''', (int(alert['timestamp']), alert_type, severity, message))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"ALERT [{severity}] {alert_type}: {message}")
    
    def get_dashboard_data(self):
        """Get data for monitoring dashboard"""
        # Recent system metrics
        system_metrics = list(self.metrics.get('system', []))[-50:]  # Last 50 entries
        
        # Service status
        service_status = {}
        for name, service in self.services.items():
            service_status[name] = {
                'status': service['status'],
                'last_check': service['last_check'],
                'avg_response_time': sum(service['response_times']) / len(service['response_times']) 
                                   if service['response_times'] else 0,
                'error_count': service['error_count'],
                'request_count': service['request_count']
            }
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts if time.time() - a['timestamp'] < 3600]  # Last hour
        
        return {
            'system_metrics': system_metrics,
            'service_status': service_status,
            'alerts': recent_alerts,
            'timestamp': time.time()
        }

# Initialize monitor
monitor = TruthMateMonitor()

# Register services (these will be detected automatically in production)
monitor.register_service('sota_service', 'http://localhost:5000')
monitor.register_service('enhanced_service', 'http://localhost:5001')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

@app.route('/metrics')
def metrics():
    """Get current metrics"""
    return jsonify(monitor.get_dashboard_data())

@app.route('/dashboard')
def dashboard():
    """Web dashboard for monitoring"""
    dashboard_html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>TruthMate Monitoring Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .metric-card { 
                border: 1px solid #ddd; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px;
                background: #f9f9f9;
            }
            .status-healthy { color: green; }
            .status-unhealthy { color: red; }
            .status-error { color: red; font-weight: bold; }
            .alert-warning { background: #fff3cd; border-color: #ffeaa7; }
            .alert-critical { background: #f8d7da; border-color: #f5c6cb; }
            .chart-container { width: 100%; height: 300px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>🔍 TruthMate Monitoring Dashboard</h1>
        
        <div id="system-status">
            <h2>System Status</h2>
            <div id="system-metrics"></div>
        </div>
        
        <div id="services-status">
            <h2>Services Status</h2>
            <div id="service-metrics"></div>
        </div>
        
        <div id="alerts-section">
            <h2>Recent Alerts</h2>
            <div id="alerts"></div>
        </div>
        
        <div id="charts-section">
            <h2>Performance Charts</h2>
            <div id="cpu-chart" class="chart-container"></div>
            <div id="memory-chart" class="chart-container"></div>
            <div id="response-time-chart" class="chart-container"></div>
        </div>
        
        <script>
            function updateDashboard() {
                fetch('/metrics')
                .then(response => response.json())
                .then(data => {
                    updateSystemMetrics(data.system_metrics);
                    updateServiceMetrics(data.service_status);
                    updateAlerts(data.alerts);
                    updateCharts(data);
                })
                .catch(error => console.error('Error:', error));
            }
            
            function updateSystemMetrics(metrics) {
                if (!metrics || metrics.length === 0) return;
                
                const latest = metrics[metrics.length - 1];
                const html = `
                    <div class="metric-card">
                        <strong>System Performance</strong><br>
                        CPU: ${latest.cpu_percent.toFixed(1)}% | 
                        Memory: ${latest.memory_percent.toFixed(1)}% | 
                        Disk: ${latest.disk_percent.toFixed(1)}%
                    </div>
                `;
                document.getElementById('system-metrics').innerHTML = html;
            }
            
            function updateServiceMetrics(services) {
                let html = '';
                for (const [name, service] of Object.entries(services)) {
                    const statusClass = `status-${service.status}`;
                    html += `
                        <div class="metric-card">
                            <strong>${name}</strong><br>
                            Status: <span class="${statusClass}">${service.status}</span><br>
                            Avg Response Time: ${service.avg_response_time.toFixed(3)}s<br>
                            Requests: ${service.request_count} | Errors: ${service.error_count}
                        </div>
                    `;
                }
                document.getElementById('service-metrics').innerHTML = html;
            }
            
            function updateAlerts(alerts) {
                let html = '';
                if (alerts.length === 0) {
                    html = '<div class="metric-card">No recent alerts</div>';
                } else {
                    alerts.forEach(alert => {
                        const alertClass = `alert-${alert.severity}`;
                        const timestamp = new Date(alert.timestamp * 1000).toLocaleString();
                        html += `
                            <div class="metric-card ${alertClass}">
                                <strong>${alert.severity.toUpperCase()}</strong> - ${alert.alert_type}<br>
                                ${alert.message}<br>
                                <small>${timestamp}</small>
                            </div>
                        `;
                    });
                }
                document.getElementById('alerts').innerHTML = html;
            }
            
            function updateCharts(data) {
                const metrics = data.system_metrics || [];
                if (metrics.length === 0) return;
                
                const timestamps = metrics.map(m => new Date(m.timestamp * 1000));
                
                // CPU Chart
                const cpuTrace = {
                    x: timestamps,
                    y: metrics.map(m => m.cpu_percent),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'CPU %',
                    line: { color: 'blue' }
                };
                Plotly.newPlot('cpu-chart', [cpuTrace], {title: 'CPU Usage'});
                
                // Memory Chart
                const memoryTrace = {
                    x: timestamps,
                    y: metrics.map(m => m.memory_percent),
                    type: 'scatter',
                    mode: 'lines', 
                    name: 'Memory %',
                    line: { color: 'green' }
                };
                Plotly.newPlot('memory-chart', [memoryTrace], {title: 'Memory Usage'});
            }
            
            // Update dashboard every 30 seconds
            updateDashboard();
            setInterval(updateDashboard, 30000);
        </script>
    </body>
    </html>
    '''
    return dashboard_html

@app.route('/alerts')
def alerts():
    """Get recent alerts"""
    recent_alerts = [a for a in monitor.alerts if time.time() - a['timestamp'] < 86400]  # Last 24 hours
    return jsonify(recent_alerts)

@app.route('/service/<service_name>/metrics')
def service_metrics(service_name):
    """Get metrics for a specific service"""
    if service_name not in monitor.services:
        return jsonify({'error': 'Service not found'}), 404
    
    # Get metrics from database
    conn = sqlite3.connect(monitor.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM service_metrics 
        WHERE service_name = ? 
        ORDER BY timestamp DESC 
        LIMIT 100
    ''', (service_name,))
    
    rows = cursor.fetchall()
    conn.close()
    
    metrics = []
    for row in rows:
        metrics.append({
            'timestamp': row[0],
            'service_name': row[1], 
            'response_time': row[2],
            'status_code': row[3],
            'error_message': row[4]
        })
    
    return jsonify(metrics)

@app.route('/ml_metrics', methods=['POST'])
def log_ml_metrics():
    """Log ML inference metrics"""
    data = request.json
    
    # Store ML metrics
    conn = sqlite3.connect(monitor.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO ml_metrics
        (timestamp, service_name, model_name, inference_time, confidence_score, verdict, input_length)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        int(time.time()),
        data.get('service_name', 'unknown'),
        data.get('model_name', 'unknown'),
        data.get('inference_time', 0),
        data.get('confidence_score', 0),
        data.get('verdict', 'unknown'),
        data.get('input_length', 0)
    ))
    
    conn.commit()
    conn.close()
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    logger.info("Starting TruthMate Monitoring Service...")
    app.run(host='0.0.0.0', port=8080, debug=False)