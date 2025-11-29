# TruthMate OS Simple Integration
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
from datetime import datetime

def analyze_url_truthmate_style(url):
    """
    TruthMate OS style URL analysis - simplified version
    Returns comprehensive analysis with description, safety score, and credibility
    """
    print(f"🕵️ TruthMate OS Analysis: {url}")
    
    try:
        # Enhanced request headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()
        
        # Parse content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract basic information
        title = soup.title.string.strip() if soup.title and soup.title.string else 'No title'
        
        # Get meta description
        meta_desc = ''
        meta_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if meta_tag:
            meta_desc = meta_tag.get('content', '').strip()
        
        # Clean content extraction
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            element.decompose()
        
        # Extract main content
        content_selectors = [
            'article', '[role="main"]', '.content', '.post-content',
            '.entry-content', '.article-body', 'main', '.main-content'
        ]
        
        main_content = ''
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = ' '.join([elem.get_text(strip=True) for elem in elements])
                break
        
        if not main_content:
            main_content = soup.get_text(strip=True)
        
        # Limit content size
        main_content = ' '.join(main_content.split())[:5000]
        
        # TruthMate OS Safety Analysis
        safety_analysis = perform_safety_analysis(url, title, main_content)
        
        # Generate description
        if meta_desc and len(meta_desc) > 20:
            description = meta_desc[:300]
        else:
            description = f"Website: {title}. " + main_content[:200] + "..."
        
        # Create summary
        summary = main_content[:800] + ("..." if len(main_content) > 800 else "")
        
        return {
            'url': url,
            'status': response.status_code,
            'page_title': title,
            'description': description,
            'summary': summary,
            'safety_score': safety_analysis['safety_score'],
            'credibility_score': safety_analysis['credibility_score'], 
            'risk_type': safety_analysis['risk_type'],
            'analysis_reason': safety_analysis['reason'],
            'content_length': len(main_content),
            'truthmate_os_style': True,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'url': url,
            'status': 0,
            'description': f'Analysis failed: {str(e)}',
            'summary': 'Unable to analyze URL',
            'safety_score': 0,
            'credibility_score': 0,
            'risk_type': 'error',
            'analysis_reason': f'Technical error: {str(e)}',
            'truthmate_os_style': True,
            'error': True
        }

def perform_safety_analysis(url, title, content):
    """Perform TruthMate OS style safety analysis"""
    safety_score = 100
    risk_factors = []
    
    # Parse URL
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # Check for suspicious TLDs
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.top', '.click']
    for tld in suspicious_tlds:
        if domain.endswith(tld):
            safety_score -= 30
            risk_factors.append(f'Suspicious TLD: {tld}')
    
    # Check for IP addresses
    if re.search(r'\d+\.\d+\.\d+\.\d+', domain):
        safety_score -= 35
        risk_factors.append('Direct IP address usage')
    
    # Content analysis
    content_lower = content.lower()
    title_lower = title.lower()
    
    # Phishing indicators
    phishing_keywords = [
        'verify your account', 'update payment', 'confirm identity',
        'suspended account', 'urgent security', 'click to verify'
    ]
    
    phishing_count = sum(1 for keyword in phishing_keywords if keyword in content_lower)
    if phishing_count > 0:
        safety_score -= phishing_count * 20
        risk_factors.append(f'{phishing_count} phishing indicator(s)')
    
    # Scam indicators  
    scam_keywords = [
        'get rich quick', 'make money fast', 'guaranteed profit',
        'miracle cure', 'doctors hate this', 'one weird trick'
    ]
    
    scam_count = sum(1 for keyword in scam_keywords if keyword in content_lower)
    if scam_count > 0:
        safety_score -= scam_count * 15
        risk_factors.append(f'{scam_count} scam indicator(s)')
    
    # Determine risk type and credibility
    safety_score = max(0, safety_score)
    
    if phishing_count > 0:
        risk_type = 'phishing'
        credibility_score = max(0, safety_score - 20)
    elif scam_count > 0:
        risk_type = 'scam'  
        credibility_score = max(0, safety_score - 15)
    elif safety_score < 60:
        risk_type = 'suspicious'
        credibility_score = safety_score
    else:
        risk_type = 'safe'
        credibility_score = min(100, safety_score + 10)
    
    # Generate reason
    if risk_factors:
        reason = f"Security analysis identified: {', '.join(risk_factors[:3])}"
    else:
        reason = "No significant security risks detected in comprehensive analysis"
    
    return {
        'safety_score': int(safety_score),
        'credibility_score': int(credibility_score),
        'risk_type': risk_type,
        'reason': reason,
        'risk_factors': risk_factors
    }