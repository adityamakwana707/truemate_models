"""
Production Deployment Script for TruthMate ML Service
Handles model serving, scaling, and monitoring
"""
import os
import sys
import argparse
import logging
import json
import time
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import signal
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
import docker
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionDeployment:
    def __init__(self):
        self.processes = {}
        self.health_checks = {}
        self.config = {}
        
    def load_deployment_config(self, config_path: str):
        """Load deployment configuration"""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                else:
                    self.config = json.load(f)
            
            logger.info("Deployment configuration loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load deployment config: {e}")
            return False
    
    def setup_environment(self):
        """Setup production environment"""
        logger.info("Setting up production environment...")
        
        # Create necessary directories
        dirs_to_create = [
            'logs',
            'models', 
            'data',
            'monitoring',
            'backups'
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Set environment variables
        env_vars = self.config.get('environment', {})
        for key, value in env_vars.items():
            os.environ[key] = str(value)
            logger.info(f"Set environment variable: {key}")
        
        # Check dependencies
        self.check_dependencies()
        
    def check_dependencies(self):
        """Check system dependencies"""
        logger.info("Checking system dependencies...")
        
        # Check Python packages
        required_packages = [
            'torch', 'transformers', 'flask', 'gunicorn',
            'numpy', 'scikit-learn', 'requests'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.info("Installing missing packages...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages, check=True)
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"GPU available: {gpu_count} devices")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    logger.info(f"GPU {i}: {gpu_name}")
            else:
                logger.info("No GPU available, using CPU")
        except:
            logger.info("PyTorch not available")
        
        # Check system resources
        memory = psutil.virtual_memory()
        logger.info(f"System memory: {memory.total / (1024**3):.1f} GB")
        logger.info(f"Available memory: {memory.available / (1024**3):.1f} GB")
        
    def deploy_flask_service(self, service_name: str, service_config: Dict):
        """Deploy Flask service with Gunicorn"""
        logger.info(f"Deploying Flask service: {service_name}")
        
        app_file = service_config.get('app_file', 'sota_app.py')
        port = service_config.get('port', 5000)
        workers = service_config.get('workers', 4)
        threads = service_config.get('threads', 2)
        timeout = service_config.get('timeout', 120)
        
        # Gunicorn command
        cmd = [
            'gunicorn',
            '--bind', f'0.0.0.0:{port}',
            '--workers', str(workers),
            '--threads', str(threads), 
            '--timeout', str(timeout),
            '--worker-class', 'sync',
            '--max-requests', '1000',
            '--max-requests-jitter', '100',
            '--preload',
            '--access-logfile', f'logs/{service_name}_access.log',
            '--error-logfile', f'logs/{service_name}_error.log',
            '--log-level', 'info',
            f'{app_file.replace(".py", "")}:app'
        ]
        
        # Environment variables for the service
        env = os.environ.copy()
        env.update(service_config.get('env_vars', {}))
        
        logger.info(f"Starting {service_name} with command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(app_file))
        )
        
        self.processes[service_name] = {
            'process': process,
            'port': port,
            'config': service_config
        }
        
        # Wait for service to start
        self.wait_for_service(service_name, port)
        
        return process
    
    def deploy_docker_service(self, service_name: str, service_config: Dict):
        """Deploy service using Docker"""
        logger.info(f"Deploying Docker service: {service_name}")
        
        try:
            client = docker.from_env()
            
            # Build or pull image
            image_name = service_config.get('image', f'{service_name}:latest')
            dockerfile_path = service_config.get('dockerfile', 'Dockerfile')
            
            if os.path.exists(dockerfile_path):
                logger.info(f"Building Docker image: {image_name}")
                client.images.build(
                    path=os.path.dirname(dockerfile_path),
                    tag=image_name,
                    rm=True
                )
            else:
                logger.info(f"Pulling Docker image: {image_name}")
                client.images.pull(image_name)
            
            # Container configuration
            container_config = {
                'image': image_name,
                'name': service_name,
                'ports': service_config.get('ports', {}),
                'environment': service_config.get('env_vars', {}),
                'volumes': service_config.get('volumes', {}),
                'detach': True,
                'remove': True
            }
            
            # GPU support
            if service_config.get('gpu', False):
                container_config['device_requests'] = [
                    docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                ]
            
            # Start container
            container = client.containers.run(**container_config)
            
            self.processes[service_name] = {
                'container': container,
                'port': list(service_config.get('ports', {}).keys())[0] if service_config.get('ports') else 5000,
                'config': service_config
            }
            
            logger.info(f"Docker service {service_name} started with ID: {container.id[:12]}")
            
        except Exception as e:
            logger.error(f"Failed to deploy Docker service {service_name}: {e}")
    
    def wait_for_service(self, service_name: str, port: int, timeout: int = 60):
        """Wait for service to become available"""
        logger.info(f"Waiting for {service_name} to start on port {port}...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f'http://localhost:{port}/health', timeout=5)
                if response.status_code == 200:
                    logger.info(f"{service_name} is ready!")
                    return True
            except:
                pass
            
            time.sleep(2)
        
        logger.error(f"{service_name} failed to start within {timeout} seconds")
        return False
    
    def setup_monitoring(self):
        """Setup monitoring and health checks"""
        logger.info("Setting up monitoring...")
        
        def health_check_worker():
            while True:
                for service_name, service_info in self.processes.items():
                    try:
                        port = service_info['port']
                        response = requests.get(
                            f'http://localhost:{port}/health', 
                            timeout=10
                        )
                        
                        status = 'healthy' if response.status_code == 200 else 'unhealthy'
                        self.health_checks[service_name] = {
                            'status': status,
                            'timestamp': time.time(),
                            'response_time': response.elapsed.total_seconds()
                        }
                        
                        if status == 'unhealthy':
                            logger.warning(f"Service {service_name} is unhealthy")
                        
                    except Exception as e:
                        logger.error(f"Health check failed for {service_name}: {e}")
                        self.health_checks[service_name] = {
                            'status': 'error',
                            'timestamp': time.time(),
                            'error': str(e)
                        }
                
                # Save health check results
                with open('monitoring/health_checks.json', 'w') as f:
                    json.dump(self.health_checks, f, indent=2)
                
                time.sleep(30)  # Check every 30 seconds
        
        # Start health check thread
        health_thread = threading.Thread(target=health_check_worker, daemon=True)
        health_thread.start()
    
    def setup_load_balancer(self):
        """Setup load balancer for multiple instances"""
        if not self.config.get('load_balancer', {}).get('enabled', False):
            return
        
        logger.info("Setting up load balancer...")
        
        # Create nginx configuration
        nginx_config = self.generate_nginx_config()
        
        with open('nginx.conf', 'w') as f:
            f.write(nginx_config)
        
        logger.info("Nginx configuration created")
    
    def generate_nginx_config(self) -> str:
        """Generate Nginx load balancer configuration"""
        services = self.config.get('services', {})
        lb_config = self.config.get('load_balancer', {})
        
        upstream_servers = []
        for service_name, service_config in services.items():
            port = service_config.get('port', 5000)
            upstream_servers.append(f"    server localhost:{port};")
        
        nginx_config = f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream truthmate_backend {{
{chr(10).join(upstream_servers)}
    }}
    
    server {{
        listen {lb_config.get('port', 80)};
        
        location / {{
            proxy_pass http://truthmate_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }}
        
        location /health {{
            access_log off;
            return 200 "healthy\\n";
            add_header Content-Type text/plain;
        }}
    }}
}}
"""
        return nginx_config
    
    def start_all_services(self):
        """Start all configured services"""
        logger.info("Starting all services...")
        
        services = self.config.get('services', {})
        
        for service_name, service_config in services.items():
            deployment_type = service_config.get('type', 'flask')
            
            try:
                if deployment_type == 'flask':
                    self.deploy_flask_service(service_name, service_config)
                elif deployment_type == 'docker':
                    self.deploy_docker_service(service_name, service_config)
                else:
                    logger.error(f"Unknown deployment type: {deployment_type}")
                
            except Exception as e:
                logger.error(f"Failed to start {service_name}: {e}")
        
        # Setup monitoring
        self.setup_monitoring()
        
        # Setup load balancer
        self.setup_load_balancer()
        
        logger.info("All services started successfully!")
    
    def stop_all_services(self):
        """Stop all running services"""
        logger.info("Stopping all services...")
        
        for service_name, service_info in self.processes.items():
            try:
                if 'process' in service_info:
                    # Flask/Gunicorn process
                    process = service_info['process']
                    process.terminate()
                    process.wait(timeout=30)
                    logger.info(f"Stopped Flask service: {service_name}")
                    
                elif 'container' in service_info:
                    # Docker container
                    container = service_info['container']
                    container.stop()
                    logger.info(f"Stopped Docker service: {service_name}")
                    
            except Exception as e:
                logger.error(f"Error stopping {service_name}: {e}")
        
        self.processes.clear()
    
    def get_service_status(self) -> Dict:
        """Get status of all services"""
        status = {
            'services': {},
            'system': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            },
            'timestamp': time.time()
        }
        
        for service_name, service_info in self.processes.items():
            service_status = {
                'port': service_info['port'],
                'config': service_info['config']
            }
            
            # Check if process/container is running
            if 'process' in service_info:
                process = service_info['process']
                service_status['running'] = process.poll() is None
                service_status['pid'] = process.pid
                
            elif 'container' in service_info:
                container = service_info['container']
                container.reload()
                service_status['running'] = container.status == 'running'
                service_status['container_id'] = container.id[:12]
            
            # Add health check info
            if service_name in self.health_checks:
                service_status['health'] = self.health_checks[service_name]
            
            status['services'][service_name] = service_status
        
        return status
    
    def run_deployment(self, config_path: str):
        """Main deployment execution"""
        if not self.load_deployment_config(config_path):
            return False
        
        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Setup environment
            self.setup_environment()
            
            # Start services
            self.start_all_services()
            
            # Keep running
            logger.info("Deployment successful! Services are running...")
            logger.info("Press Ctrl+C to stop all services")
            
            while True:
                time.sleep(10)
                
                # Print status every 5 minutes
                if int(time.time()) % 300 == 0:
                    status = self.get_service_status()
                    logger.info(f"System status: CPU {status['system']['cpu_usage']}%, "
                              f"Memory {status['system']['memory_usage']}%")
        
        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
        except Exception as e:
            logger.error(f"Deployment error: {e}")
        finally:
            self.stop_all_services()
        
        return True
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_all_services()
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='TruthMate Production Deployment')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Deployment configuration file')
    parser.add_argument('--status', action='store_true',
                       help='Show status of running services')
    parser.add_argument('--stop', action='store_true',
                       help='Stop all services')
    
    args = parser.parse_args()
    
    deployment = ProductionDeployment()
    
    if args.status:
        if deployment.load_deployment_config(args.config):
            status = deployment.get_service_status()
            print(json.dumps(status, indent=2))
    
    elif args.stop:
        if deployment.load_deployment_config(args.config):
            deployment.stop_all_services()
    
    else:
        # Run deployment
        success = deployment.run_deployment(args.config)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

# Example usage:
# python deploy.py --config production_config.yaml
# python deploy.py --config production_config.yaml --status
# python deploy.py --config production_config.yaml --stop