import re
from typing import Dict, List
from datetime import datetime
import json

class NLPComplianceEngine:
    
    def __init__(self):
        self.risk_keywords = {
            'high': ['fraud', 'embezzlement', 'misappropriation', 'criminal', 'violation', 
                     'breach', 'failure', 'deficiency', 'inadequate', 'severe', 'critical',
                     'sanctions', 'penalty', 'enforcement', 'prosecution'],
            'medium': ['concern', 'weakness', 'issue', 'problem', 'risk', 'exposure',
                      'non-compliance', 'deviation', 'discrepancy', 'irregularity'],
            'low': ['note', 'observation', 'recommendation', 'suggestion', 'improvement',
                    'enhancement', 'consider', 'review']
        }
        
        self.positive_keywords = ['compliant', 'satisfactory', 'adequate', 'strong',
                                  'robust', 'effective', 'sound', 'well-managed']
        
        self.compliance_topics = {
            'capital': ['capital', 'adequacy', 'solvency', 'equity', 'tier 1', 'tier 2'],
            'liquidity': ['liquidity', 'cash', 'liquid assets', 'funding', 'deposits'],
            'credit_risk': ['credit risk', 'loans', 'npl', 'non-performing', 'provisions'],
            'governance': ['governance', 'board', 'management', 'oversight', 'controls'],
            'aml': ['anti-money laundering', 'aml', 'kyc', 'suspicious transactions'],
            'reporting': ['reporting', 'disclosure', 'transparency', 'returns']
        }
    
    def analyze_document(self, document_text: str, document_type: str = 'audit_report') -> Dict:
        if not document_text or len(document_text.strip()) == 0:
            return {
                'sentiment_score': 0.5,
                'risk_signals': [],
                'key_topics': [],
                'entities_extracted': [],
                'compliance_score': 3.0
            }
        
        sentiment = self._analyze_sentiment(document_text)
        
        risk_signals = self._extract_risk_signals(document_text)
        
        topics = self._identify_topics(document_text)
        
        entities = self._extract_entities(document_text)
        
        compliance_score = self._calculate_compliance_score(sentiment, risk_signals)
        
        return {
            'sentiment_score': sentiment,
            'risk_signals': risk_signals,
            'key_topics': topics,
            'entities_extracted': entities,
            'compliance_score': compliance_score
        }
    
    def _analyze_sentiment(self, text: str) -> float:
        text_lower = text.lower()
        
        risk_count = 0
        positive_count = 0
        
        for severity, keywords in self.risk_keywords.items():
            for keyword in keywords:
                count = text_lower.count(keyword)
                if severity == 'high':
                    risk_count += count * 3
                elif severity == 'medium':
                    risk_count += count * 2
                else:
                    risk_count += count
        
        for keyword in self.positive_keywords:
            positive_count += text_lower.count(keyword) * 2
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.5
        
        risk_density = risk_count / max(total_words / 100, 1)
        positive_density = positive_count / max(total_words / 100, 1)
        
        sentiment = 0.5 + (positive_density - risk_density) / 20
        sentiment = max(0.0, min(1.0, sentiment))
        
        return round(sentiment, 4)
    
    def _extract_risk_signals(self, text: str) -> List[Dict]:
        risk_signals = []
        text_lower = text.lower()
        sentences = text.split('.')
        
        for idx, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            for severity, keywords in self.risk_keywords.items():
                for keyword in keywords:
                    if keyword in sentence_lower:
                        context = sentence.strip()[:200]
                        
                        risk_signals.append({
                            'severity': severity,
                            'keyword': keyword,
                            'context': context,
                            'sentence_index': idx
                        })
        
        return risk_signals
    
    def _identify_topics(self, text: str) -> List[Dict]:
        text_lower = text.lower()
        identified_topics = []
        
        for topic_name, keywords in self.compliance_topics.items():
            mentions = 0
            for keyword in keywords:
                mentions += text_lower.count(keyword)
            
            if mentions > 0:
                identified_topics.append({
                    'topic': topic_name,
                    'mentions': mentions,
                    'relevance': min(mentions / 5, 1.0)
                })
        
        identified_topics.sort(key=lambda x: x['mentions'], reverse=True)
        
        return identified_topics
    
    def _extract_entities(self, text: str) -> List[Dict]:
        entities = []
        
        amount_pattern = r'\$[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:million|billion|USD|ZWL)'
        amounts = re.findall(amount_pattern, text, re.IGNORECASE)
        for amount in amounts[:10]:
            entities.append({
                'type': 'monetary_amount',
                'value': amount.strip()
            })
        
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        for date in dates[:5]:
            entities.append({
                'type': 'date',
                'value': date.strip()
            })
        
        percentage_pattern = r'\d+(?:\.\d+)?%'
        percentages = re.findall(percentage_pattern, text)
        for pct in percentages[:10]:
            entities.append({
                'type': 'percentage',
                'value': pct
            })
        
        return entities
    
    def _calculate_compliance_score(self, sentiment: float, risk_signals: List[Dict]) -> float:
        base_score = sentiment * 5
        
        high_risk_count = sum(1 for signal in risk_signals if signal['severity'] == 'high')
        medium_risk_count = sum(1 for signal in risk_signals if signal['severity'] == 'medium')
        
        penalty = (high_risk_count * 0.5) + (medium_risk_count * 0.25)
        
        compliance_score = base_score - penalty
        compliance_score = max(1.0, min(5.0, compliance_score))
        
        return round(compliance_score, 2)
    
    def analyze_media_mentions(self, mentions: List[str]) -> Dict:
        if not mentions:
            return {
                'total_mentions': 0,
                'negative_mentions': 0,
                'positive_mentions': 0,
                'neutral_mentions': 0,
                'overall_sentiment': 'neutral',
                'risk_level': 'low'
            }
        
        negative_count = 0
        positive_count = 0
        neutral_count = 0
        
        for mention in mentions:
            sentiment = self._analyze_sentiment(mention)
            
            if sentiment < 0.4:
                negative_count += 1
            elif sentiment > 0.6:
                positive_count += 1
            else:
                neutral_count += 1
        
        total = len(mentions)
        
        if negative_count > total * 0.5:
            overall_sentiment = 'negative'
            risk_level = 'high'
        elif negative_count > total * 0.3:
            overall_sentiment = 'mixed_negative'
            risk_level = 'medium'
        elif positive_count > total * 0.5:
            overall_sentiment = 'positive'
            risk_level = 'low'
        else:
            overall_sentiment = 'neutral'
            risk_level = 'low'
        
        return {
            'total_mentions': total,
            'negative_mentions': negative_count,
            'positive_mentions': positive_count,
            'neutral_mentions': neutral_count,
            'overall_sentiment': overall_sentiment,
            'risk_level': risk_level
        }
    
    def generate_compliance_report(self, analysis_results: Dict, bank_name: str) -> str:
        report_lines = []
        
        report_lines.append(f"=== NLP Compliance Analysis Report ===")
        report_lines.append(f"Bank: {bank_name}")
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        sentiment = analysis_results.get('sentiment_score', 0.5)
        report_lines.append(f"Sentiment Score: {sentiment:.2f} (0=Negative, 1=Positive)")
        
        compliance_score = analysis_results.get('compliance_score', 3.0)
        report_lines.append(f"Compliance Score: {compliance_score:.2f} / 5.0")
        
        status = 'Strong' if compliance_score >= 4 else 'Satisfactory' if compliance_score >= 3 else 'Needs Improvement' if compliance_score >= 2 else 'Poor'
        report_lines.append(f"Overall Status: {status}")
        report_lines.append("")
        
        risk_signals = analysis_results.get('risk_signals', [])
        if risk_signals:
            report_lines.append(f"Risk Signals Detected: {len(risk_signals)}")
            
            high_risks = [s for s in risk_signals if s['severity'] == 'high']
            if high_risks:
                report_lines.append(f"\nHigh Severity Risks ({len(high_risks)}):")
                for risk in high_risks[:5]:
                    report_lines.append(f"  • {risk['keyword']}: {risk['context'][:100]}...")
        
        topics = analysis_results.get('key_topics', [])
        if topics:
            report_lines.append(f"\nKey Topics Identified:")
            for topic in topics[:5]:
                report_lines.append(f"  • {topic['topic']}: {topic['mentions']} mentions")
        
        return "\n".join(report_lines)
