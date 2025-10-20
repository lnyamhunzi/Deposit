from sqlalchemy.orm import Session
from sqlalchemy import func, extract, case, text
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import tempfile
import os

class ComprehensiveReportingService:
    def __init__(self, db: Session):
        self.db = db

    async def generate_custom_report(self, report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate customizable reports with various visualization options"""
        
        report_type = report_config.get('type', 'standard')
        
        if report_type == 'financial_summary':
            return await self._generate_financial_summary_report(report_config)
        elif report_type == 'risk_analysis':
            return await self._generate_risk_analysis_report(report_config)
        elif report_type == 'compliance_tracking':
            return await self._generate_compliance_report(report_config)
        elif report_type == 'performance_metrics':
            return await self._generate_performance_report(report_config)
        else:
            return await self._generate_standard_report(report_config)
    
    async def _generate_financial_summary_report(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive financial summary report"""
        
        date_range = config.get('date_range', {})
        institution_id = config.get('institution_id')
        
        # Get financial data
        financial_data = await self._get_financial_data(date_range, institution_id)
        
        # Create visualizations
        charts = await self._create_financial_charts(financial_data)
        
        # Generate insights
        insights = await self._generate_financial_insights(financial_data)
        
        return {
            'report_type': 'financial_summary',
            'generated_at': datetime.utcnow().isoformat(),
            'date_range': date_range,
            'summary_metrics': financial_data['summary_metrics'],
            'charts': charts,
            'insights': insights,
            'recommendations': await self._generate_financial_recommendations(financial_data),
            'data_snapshot': financial_data['detailed_data'][:100]  # First 100 records
        }
    
    async def _get_financial_data(self, date_range: Dict[str, str], institution_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve financial data for reporting"""
        
        # This would query actual financial data from the database
        # For now, return mock data structure
        
        return {
            'summary_metrics': {
                'total_deposits': 1500000000.00,
                'total_premiums': 2250000.00,
                'total_insured': 1200000000.00,
                'collection_rate': 94.5,
                'average_account_balance': 25000.00,
                'growth_rate': 8.2
            },
            'trend_data': {
                'deposit_growth': [1200000, 1250000, 1300000, 1350000, 1400000, 1450000, 1500000],
                'premium_collection': [1800, 1900, 2000, 2100, 2200, 2250, 2300],
                'coverage_ratio': [78, 79, 80, 81, 82, 83, 84]
            },
            'breakdowns': {
                'by_institution_type': {
                    'Commercial Banks': 65,
                    'Microfinance': 20,
                    'Credit Unions': 15
                },
                'by_deposit_size': {
                    'Small (<$10k)': 45,
                    'Medium ($10k-$100k)': 35,
                    'Large (>$100k)': 20
                }
            },
            'detailed_data': [
                {
                    'period': '2024-Q1',
                    'deposits': 1450000,
                    'premiums': 2200,
                    'insured_amount': 1160000,
                    'growth': 7.8
                }
                # ... more records
            ]
        }
    
    async def _create_financial_charts(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create financial charts and visualizations"""
        
        charts = {}
        
        # Trend line chart
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=list(range(1, 8)),
            y=financial_data['trend_data']['deposit_growth'],
            mode='lines+markers',
            name='Total Deposits',
            line=dict(color='#1f77b4')
        ))
        trend_fig.add_trace(go.Scatter(
            x=list(range(1, 8)),
            y=financial_data['trend_data']['premium_collection'],
            mode='lines+markers',
            name='Premium Collection',
            line=dict(color='#ff7f0e'),
            yaxis='y2'
        ))
        
        trend_fig.update_layout(
            title='Deposit and Premium Trends',
            xaxis_title='Period',
            yaxis=dict(title='Deposits (Millions)'),
            yaxis2=dict(title='Premiums (Thousands)', overlaying='y', side='right'),
            template='plotly_white'
        )
        
        charts['trend_chart'] = trend_fig.to_json()
        
        # Pie chart for institution breakdown
        pie_fig = px.pie(
            values=list(financial_data['breakdowns']['by_institution_type'].values()),
            names=list(financial_data['breakdowns']['by_institution_type'].keys()),
            title='Deposit Distribution by Institution Type'
        )
        charts['institution_breakdown'] = pie_fig.to_json()
        
        # Bar chart for deposit size distribution
        bar_fig = px.bar(
            x=list(financial_data['breakdowns']['by_deposit_size'].keys()),
            y=list(financial_data['breakdowns']['by_deposit_size'].values()),
            title='Deposit Size Distribution',
            color=list(financial_data['breakdowns']['by_deposit_size'].keys()),
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        charts['deposit_size_chart'] = bar_fig.to_json()
        
        return charts
    
    async def _generate_financial_insights(self, financial_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate financial insights from data"""
        
        insights = []
        metrics = financial_data['summary_metrics']
        
        # Growth insight
        if metrics['growth_rate'] > 10:
            insights.append({
                'type': 'positive',
                'category': 'growth',
                'title': 'Strong Deposit Growth',
                'description': f"Deposits growing at {metrics['growth_rate']}% annually",
                'impact': 'high',
                'recommendation': 'Continue current growth strategies'
            })
        elif metrics['growth_rate'] < 5:
            insights.append({
                'type': 'warning',
                'category': 'growth',
                'title': 'Slow Deposit Growth',
                'description': f"Deposit growth rate of {metrics['growth_rate']}% below target",
                'impact': 'medium',
                'recommendation': 'Review deposit acquisition strategies'
            })
        
        # Collection efficiency insight
        if metrics['collection_rate'] > 95:
            insights.append({
                'type': 'positive',
                'category': 'collections',
                'title': 'Excellent Collection Efficiency',
                'description': f"Premium collection rate of {metrics['collection_rate']}%",
                'impact': 'high',
                'recommendation': 'Maintain current collection processes'
            })
        
        return insights
    
    async def _generate_financial_recommendations(self, financial_data: Dict[str, Any]) -> List[str]:
        """Generate financial recommendations"""
        
        recommendations = []
        metrics = financial_data['summary_metrics']
        
        if metrics['growth_rate'] < 8:
            recommendations.append("Implement targeted deposit growth initiatives")
        
        if metrics['collection_rate'] < 95:
            recommendations.append("Enhance premium collection processes and follow-up procedures")
        
        recommendations.append("Conduct regular financial health assessments of member institutions")
        recommendations.append("Diversify investment portfolio to maximize returns on premiums")
        
        return recommendations
    
    async def export_report(self, report_data: Dict[str, Any], format: str = 'pdf') -> Dict[str, Any]:
        """Export report in various formats"""
        
        if format == 'pdf':
            return await self._export_pdf_report(report_data)
        elif format == 'excel':
            return await self._export_excel_report(report_data)
        elif format == 'html':
            return await self._export_html_report(report_data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
    
    async def _export_pdf_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export report as PDF"""
        try:
            # This would use a PDF generation library like ReportLab or WeasyPrint
            # For now, return a placeholder
            
            filename = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            return {
                'filename': filename,
                'content_type': 'application/pdf',
                'message': 'PDF export would be implemented with a PDF generation library',
                'placeholder': report_data
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")
    
    async def create_dashboard(self, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create interactive dashboard with multiple visualizations"""
        
        widgets = []
        
        for widget_config in dashboard_config.get('widgets', []):
            widget = await self._create_dashboard_widget(widget_config)
            widgets.append(widget)
        
        return {
            'dashboard_id': f"dashboard_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'title': dashboard_config.get('title', 'Custom Dashboard'),
            'widgets': widgets,
            'layout': dashboard_config.get('layout', 'grid'),
            'refresh_interval': dashboard_config.get('refresh_interval', 300),
            'created_at': datetime.utcnow().isoformat()
        }
    
    async def _create_dashboard_widget(self, widget_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create individual dashboard widget"""
        
        widget_type = widget_config.get('type', 'metric')
        
        if widget_type == 'metric':
            return await self._create_metric_widget(widget_config)
        elif widget_type == 'chart':
            return await self._create_chart_widget(widget_config)
        elif widget_type == 'table':
            return await self._create_table_widget(widget_config)
        else:
            return await self._create_metric_widget(widget_config)
    
    async def _create_metric_widget(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create metric display widget"""
        
        metric_data = await self._get_metric_data(config.get('metric_name'))
        
        return {
            'type': 'metric',
            'title': config.get('title', 'Metric'),
            'value': metric_data['current'],
            'previous_value': metric_data.get('previous'),
            'change': metric_data.get('change'),
            'change_percentage': metric_data.get('change_percentage'),
            'trend': metric_data.get('trend', 'neutral'),
            'format': config.get('format', 'number'),
            'size': config.get('size', 'medium')
        }
    
    async def _get_metric_data(self, metric_name: str) -> Dict[str, Any]:
        """Retrieve metric data for dashboard"""
        
        # Mock data - would query actual metrics
        metrics_data = {
            'total_deposits': {
                'current': 1500000000,
                'previous': 1450000000,
                'change': 50000000,
                'change_percentage': 3.45,
                'trend': 'up'
            },
            'collection_rate': {
                'current': 94.5,
                'previous': 93.8,
                'change': 0.7,
                'change_percentage': 0.75,
                'trend': 'up'
            },
            'risk_exposure': {
                'current': 12.3,
                'previous': 11.8,
                'change': 0.5,
                'change_percentage': 4.24,
                'trend': 'down'
            }
        }
        
        return metrics_data.get(metric_name, {'current': 0, 'trend': 'neutral'})