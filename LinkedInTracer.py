import json
import networkx as nx
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
from datetime import datetime
import logging
import requests
import time
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from tavily import TavilyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    id: str
    label: str
    node_type: str
    properties: Dict[str, Any]
    
@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph"""
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any]

@dataclass
class KnowledgeGraph:
    """Complete knowledge graph structure"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any]

@dataclass
class Experience:
    title: str
    company: str
    duration: str
    location: str
    description: str
    employment_type: str

@dataclass
class Education:
    institution: str
    degree: str
    field_of_study: str
    duration: str
    grade: str
    activities: str

@dataclass
class Certification:
    name: str
    issuing_organization: str
    issue_date: str
    expiration_date: str
    credential_id: str
    credential_url: str

@dataclass
class LinkedInProfile:
    name: str
    headline: str
    location: str
    about: str
    profile_url: str
    profile_image_url: str
    background_image_url: str
    connections: str
    followers: str
    experience: List[Experience]
    education: List[Education]
    skills: List[str]
    certifications: List[Certification]
    languages: List[str]
    achievements: List[str]
    projects: List[str]
    publications: List[str]
    contact_info: Dict[str, str]

class LinkedInKnowledgeGraphBuilder:
    """Converts LinkedIn profile data into a structured knowledge graph"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes = []
        self.edges = []
        
    def build_knowledge_graph(self, profile_data: Dict) -> KnowledgeGraph:
        """Main method to build knowledge graph from LinkedIn profile data"""
        logger.info("Building knowledge graph from LinkedIn profile data...")
        
        # Create person node (central node)
        person_id = self._create_person_node(profile_data)
        
        # Add experience nodes and relationships
        self._add_experience_nodes(profile_data, person_id)
        
        # Add education nodes and relationships
        self._add_education_nodes(profile_data, person_id)
        
        # Add skill nodes and relationships
        self._add_skill_nodes(profile_data, person_id)
        
        # Add certification nodes and relationships
        self._add_certification_nodes(profile_data, person_id)
        
        # Add project nodes and relationships
        self._add_project_nodes(profile_data, person_id)
        
        # Add location node and relationship
        self._add_location_node(profile_data, person_id)
        
        # Add contact information nodes
        self._add_contact_nodes(profile_data, person_id)
        
        # Infer additional relationships
        self._infer_relationships()
        
        # Create final knowledge graph
        kg = KnowledgeGraph(
            nodes=self.nodes,
            edges=self.edges,
            metadata={
                "created_at": datetime.now().isoformat(),
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "profile_url": profile_data.get("profile_url", ""),
                "data_quality_score": self._calculate_data_quality_score(profile_data)
            }
        )
        
        logger.info(f"Knowledge graph created with {len(self.nodes)} nodes and {len(self.edges)} edges")
        return kg
    
    def _create_person_node(self, profile_data: Dict) -> str:
        """Create the central person node"""
        person_id = f"person_{self._clean_id(profile_data.get('name', 'unknown'))}"
        
        person_node = GraphNode(
            id=person_id,
            label=profile_data.get('name', 'Unknown Person'),
            node_type="Person",
            properties={
                "name": profile_data.get('name', 'Unknown'),
                "headline": profile_data.get('headline', 'Not specified'),
                "about": profile_data.get('about', 'No summary available'),
                "profile_url": profile_data.get('profile_url', ''),
                "connections": profile_data.get('connections', 'Not specified'),
                "followers": profile_data.get('followers', 'Not specified'),
                "profile_image_url": profile_data.get('profile_image_url', ''),
                "is_central_node": True
            }
        )
        
        self.nodes.append(person_node)
        self.graph.add_node(person_id, **asdict(person_node))
        return person_id
    
    def _add_experience_nodes(self, profile_data: Dict, person_id: str):
        """Add work experience nodes and relationships"""
        experiences = profile_data.get('experience', [])
        
        for i, exp in enumerate(experiences):
            if not exp.get('title') or exp['title'] == 'Not specified':
                continue
                
            # Create job position node
            job_id = f"job_{i}_{self._clean_id(exp.get('title', ''))}"
            job_node = GraphNode(
                id=job_id,
                label=exp.get('title', 'Unknown Position'),
                node_type="JobPosition",
                properties={
                    "title": exp.get('title', ''),
                    "description": exp.get('description', ''),
                    "duration": exp.get('duration', 'Not specified'),
                    "employment_type": exp.get('employment_type', 'Not specified'),
                    "start_date": self._extract_start_date(exp.get('duration', '')),
                    "end_date": self._extract_end_date(exp.get('duration', '')),
                    "is_current": self._is_current_position(exp.get('duration', ''))
                }
            )
            self.nodes.append(job_node)
            self.graph.add_node(job_id, **asdict(job_node))
            
            # Create company node if specified
            if exp.get('company') and exp['company'] != 'Not specified':
                company_id = f"company_{self._clean_id(exp['company'])}"
                
                # Check if company node already exists
                existing_company = self._find_node_by_id(company_id)
                if not existing_company:
                    company_node = GraphNode(
                        id=company_id,
                        label=exp['company'],
                        node_type="Company",
                        properties={
                            "name": exp['company'],
                            "industry": self._infer_industry(exp['company'], exp.get('title', '')),
                            "location": exp.get('location', 'Not specified')
                        }
                    )
                    self.nodes.append(company_node)
                    self.graph.add_node(company_id, **asdict(company_node))
                
                # Add relationships
                self._add_edge(person_id, job_id, "WORKED_AS", {
                    "duration": exp.get('duration', ''),
                    "location": exp.get('location', '')
                })
                
                self._add_edge(job_id, company_id, "EMPLOYED_BY", {
                    "duration": exp.get('duration', '')
                })
            else:
                # Direct relationship if no company specified
                self._add_edge(person_id, job_id, "HAS_EXPERIENCE", {
                    "duration": exp.get('duration', ''),
                    "location": exp.get('location', '')
                })
    
    def _add_education_nodes(self, profile_data: Dict, person_id: str):
        """Add education nodes and relationships"""
        educations = profile_data.get('education', [])
        
        for i, edu in enumerate(educations):
            if not edu.get('institution') or edu['institution'] == 'Not specified':
                continue
                
            # Create institution node
            institution_id = f"institution_{self._clean_id(edu['institution'])}"
            
            existing_institution = self._find_node_by_id(institution_id)
            if not existing_institution:
                institution_node = GraphNode(
                    id=institution_id,
                    label=edu['institution'],
                    node_type="EducationalInstitution",
                    properties={
                        "name": edu['institution'],
                        "type": self._classify_institution_type(edu['institution'])
                    }
                )
                self.nodes.append(institution_node)
                self.graph.add_node(institution_id, **asdict(institution_node))
            
            # Create degree node if specified
            if edu.get('degree') and edu['degree'] != 'Not specified':
                degree_id = f"degree_{i}_{self._clean_id(edu['degree'])}"
                degree_node = GraphNode(
                    id=degree_id,
                    label=f"{edu['degree']} in {edu.get('field_of_study', 'Unspecified')}",
                    node_type="Degree",
                    properties={
                        "degree_type": edu['degree'],
                        "field_of_study": edu.get('field_of_study', 'Not specified'),
                        "duration": edu.get('duration', 'Not specified'),
                        "grade": edu.get('grade', 'Not specified'),
                        "activities": edu.get('activities', 'Not specified')
                    }
                )
                self.nodes.append(degree_node)
                self.graph.add_node(degree_id, **asdict(degree_node))
                
                # Add relationships
                self._add_edge(person_id, degree_id, "EARNED_DEGREE", {
                    "duration": edu.get('duration', ''),
                    "grade": edu.get('grade', '')
                })
                
                self._add_edge(degree_id, institution_id, "AWARDED_BY", {})
            else:
                # Direct relationship if no degree specified
                self._add_edge(person_id, institution_id, "STUDIED_AT", {
                    "duration": edu.get('duration', ''),
                    "field": edu.get('field_of_study', '')
                })
    
    def _add_skill_nodes(self, profile_data: Dict, person_id: str):
        """Add skill nodes and relationships"""
        skills = profile_data.get('skills', [])
        
        for skill in skills:
            if not skill or skill.strip() == '':
                continue
                
            skill_id = f"skill_{self._clean_id(skill)}"
            
            existing_skill = self._find_node_by_id(skill_id)
            if not existing_skill:
                skill_node = GraphNode(
                    id=skill_id,
                    label=skill,
                    node_type="Skill",
                    properties={
                        "name": skill,
                        "category": self._categorize_skill(skill),
                        "is_technical": self._is_technical_skill(skill)
                    }
                )
                self.nodes.append(skill_node)
                self.graph.add_node(skill_id, **asdict(skill_node))
            
            self._add_edge(person_id, skill_id, "HAS_SKILL", {
                "proficiency": "Not specified"
            })
    
    def _add_certification_nodes(self, profile_data: Dict, person_id: str):
        """Add certification nodes and relationships"""
        certifications = profile_data.get('certifications', [])
        
        for i, cert in enumerate(certifications):
            if not cert.get('name'):
                continue
                
            cert_id = f"cert_{i}_{self._clean_id(cert['name'])}"
            cert_node = GraphNode(
                id=cert_id,
                label=cert['name'],
                node_type="Certification",
                properties={
                    "name": cert['name'],
                    "issuing_organization": cert.get('issuing_organization', 'Not specified'),
                    "issue_date": cert.get('issue_date', 'Not specified'),
                    "expiration_date": cert.get('expiration_date', 'Not specified'),
                    "credential_id": cert.get('credential_id', 'Not specified'),
                    "credential_url": cert.get('credential_url', 'Not specified'),
                    "is_active": self._is_certification_active(cert.get('expiration_date', ''))
                }
            )
            self.nodes.append(cert_node)
            self.graph.add_node(cert_id, **asdict(cert_node))
            
            # Create issuing organization node if specified
            if cert.get('issuing_organization') and cert['issuing_organization'] != 'Not specified':
                org_id = f"org_{self._clean_id(cert['issuing_organization'])}"
                
                existing_org = self._find_node_by_id(org_id)
                if not existing_org:
                    org_node = GraphNode(
                        id=org_id,
                        label=cert['issuing_organization'],
                        node_type="Organization",
                        properties={
                            "name": cert['issuing_organization'],
                            "type": "Certification Authority"
                        }
                    )
                    self.nodes.append(org_node)
                    self.graph.add_node(org_id, **asdict(org_node))
                
                self._add_edge(cert_id, org_id, "ISSUED_BY", {})
            
            self._add_edge(person_id, cert_id, "HOLDS_CERTIFICATION", {
                "issue_date": cert.get('issue_date', ''),
                "status": "active" if self._is_certification_active(cert.get('expiration_date', '')) else "expired"
            })
    
    def _add_project_nodes(self, profile_data: Dict, person_id: str):
        """Add project nodes and relationships"""
        projects = profile_data.get('projects', [])
        
        for i, project in enumerate(projects):
            if not project or project.strip() == '':
                continue
                
            project_id = f"project_{i}_{self._clean_id(project[:20])}"
            project_node = GraphNode(
                id=project_id,
                label=project[:50] + "..." if len(project) > 50 else project,
                node_type="Project",
                properties={
                    "description": project,
                    "technologies": self._extract_technologies(project),
                    "domain": self._infer_project_domain(project)
                }
            )
            self.nodes.append(project_node)
            self.graph.add_node(project_id, **asdict(project_node))
            
            self._add_edge(person_id, project_id, "WORKED_ON", {})
    
    def _add_location_node(self, profile_data: Dict, person_id: str):
        """Add location node and relationship"""
        location = profile_data.get('location', '')
        if location and location != 'Not specified':
            location_id = f"location_{self._clean_id(location)}"
            
            existing_location = self._find_node_by_id(location_id)
            if not existing_location:
                location_node = GraphNode(
                    id=location_id,
                    label=location,
                    node_type="Location",
                    properties={
                        "name": location,
                        "type": self._classify_location_type(location)
                    }
                )
                self.nodes.append(location_node)
                self.graph.add_node(location_id, **asdict(location_node))
            
            self._add_edge(person_id, location_id, "LOCATED_IN", {})
    
    def _add_contact_nodes(self, profile_data: Dict, person_id: str):
        """Add contact information nodes"""
        contact_info = profile_data.get('contact_info', {})
        
        for contact_type, contact_value in contact_info.items():
            if not contact_value:
                continue
                
            contact_id = f"contact_{contact_type}_{self._clean_id(contact_value)}"
            contact_node = GraphNode(
                id=contact_id,
                label=contact_value,
                node_type="ContactInfo",
                properties={
                    "value": contact_value,
                    "type": contact_type,
                    "is_public": True
                }
            )
            self.nodes.append(contact_node)
            self.graph.add_node(contact_id, **asdict(contact_node))
            
            self._add_edge(person_id, contact_id, "HAS_CONTACT", {
                "contact_type": contact_type
            })
    
    def _infer_relationships(self):
        """Infer additional relationships between nodes"""
        # Connect skills to job positions based on relevance
        self._connect_skills_to_jobs()
        
        # Connect education to career progression
        self._connect_education_to_career()
        
        # Connect certifications to skills
        self._connect_certifications_to_skills()
    
    def _connect_skills_to_jobs(self):
        """Connect skills to relevant job positions"""
        skill_nodes = [n for n in self.nodes if n.node_type == "Skill"]
        job_nodes = [n for n in self.nodes if n.node_type == "JobPosition"]
        
        for skill_node in skill_nodes:
            for job_node in job_nodes:
                if self._is_skill_relevant_to_job(skill_node.properties['name'], 
                                                job_node.properties.get('title', ''),
                                                job_node.properties.get('description', '')):
                    self._add_edge(job_node.id, skill_node.id, "REQUIRES_SKILL", {
                        "relevance": "inferred"
                    })
    
    def _connect_education_to_career(self):
        """Connect education to career progression"""
        education_nodes = [n for n in self.nodes if n.node_type == "Degree"]
        job_nodes = [n for n in self.nodes if n.node_type == "JobPosition"]
        
        for edu_node in education_nodes:
            field = edu_node.properties.get('field_of_study', '').lower()
            for job_node in job_nodes:
                job_title = job_node.properties.get('title', '').lower()
                if self._is_education_relevant_to_job(field, job_title):
                    self._add_edge(edu_node.id, job_node.id, "PREPARED_FOR", {
                        "relevance": "inferred"
                    })
    
    def _connect_certifications_to_skills(self):
        """Connect certifications to related skills"""
        cert_nodes = [n for n in self.nodes if n.node_type == "Certification"]
        skill_nodes = [n for n in self.nodes if n.node_type == "Skill"]
        
        for cert_node in cert_nodes:
            cert_name = cert_node.properties['name'].lower()
            for skill_node in skill_nodes:
                skill_name = skill_node.properties['name'].lower()
                if skill_name in cert_name or self._are_related_cert_skill(cert_name, skill_name):
                    self._add_edge(cert_node.id, skill_node.id, "VALIDATES_SKILL", {
                        "confidence": "medium"
                    })
    
    def _add_edge(self, source: str, target: str, relationship: str, properties: Dict = None):
        """Add an edge to the knowledge graph"""
        if properties is None:
            properties = {}
            
        edge = GraphEdge(
            source=source,
            target=target,
            relationship=relationship,
            properties=properties
        )
        self.edges.append(edge)
        self.graph.add_edge(source, target, relationship=relationship, **properties)
    
    def _find_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Find a node by its ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def _clean_id(self, text: str) -> str:
        """Clean text to create valid node IDs"""
        if not text:
            return "unknown"
        return re.sub(r'[^a-zA-Z0-9_]', '_', text.lower()).strip('_')
    
    def _extract_start_date(self, duration: str) -> str:
        """Extract start date from duration string"""
        if not duration or duration == 'Not specified':
            return 'Not specified'
        
        # Simple pattern matching for dates
        patterns = [
            r'(\d{4})\s*[-–]\s*\d{4}',
            r'(\d{4})\s*[-–]\s*Present',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, duration, re.IGNORECASE)
            if match:
                return match.group(1) if len(match.groups()) == 1 else f"{match.group(1)} {match.group(2)}"
        
        return 'Not specified'
    
    def _extract_end_date(self, duration: str) -> str:
        """Extract end date from duration string"""
        if not duration or duration == 'Not specified':
            return 'Not specified'
        
        if 'present' in duration.lower():
            return 'Present'
        
        # Look for end date patterns
        match = re.search(r'[-–]\s*(\d{4})', duration)
        if match:
            return match.group(1)
        
        return 'Not specified'
    
    def _is_current_position(self, duration: str) -> bool:
        """Check if position is current"""
        return 'present' in duration.lower() if duration else False
    
    def _infer_industry(self, company: str, job_title: str) -> str:
        """Infer industry from company name and job title"""
        tech_keywords = ['software', 'technology', 'tech', 'digital', 'data', 'ai', 'ml']
        finance_keywords = ['bank', 'financial', 'investment', 'capital']
        healthcare_keywords = ['health', 'medical', 'hospital', 'pharma']
        
        text = f"{company} {job_title}".lower()
        
        if any(keyword in text for keyword in tech_keywords):
            return 'Technology'
        elif any(keyword in text for keyword in finance_keywords):
            return 'Finance'
        elif any(keyword in text for keyword in healthcare_keywords):
            return 'Healthcare'
        else:
            return 'Other'
    
    def _classify_institution_type(self, institution: str) -> str:
        """Classify type of educational institution"""
        inst_lower = institution.lower()
        if 'university' in inst_lower:
            return 'University'
        elif 'college' in inst_lower:
            return 'College'
        elif 'institute' in inst_lower:
            return 'Institute'
        elif 'school' in inst_lower:
            return 'School'
        else:
            return 'Educational Institution'
    
    def _categorize_skill(self, skill: str) -> str:
        """Categorize skill type"""
        tech_skills = ['python', 'java', 'javascript', 'react', 'aws', 'docker', 'sql', 'ai', 'ml']
        soft_skills = ['leadership', 'communication', 'management', 'teamwork']
        
        skill_lower = skill.lower()
        
        if any(tech in skill_lower for tech in tech_skills):
            return 'Technical'
        elif any(soft in skill_lower for soft in soft_skills):
            return 'Soft Skill'
        else:
            return 'Professional'
    
    def _is_technical_skill(self, skill: str) -> bool:
        """Check if skill is technical"""
        return self._categorize_skill(skill) == 'Technical'
    
    def _is_certification_active(self, expiration_date: str) -> bool:
        """Check if certification is still active"""
        if not expiration_date or expiration_date == 'Not specified':
            return True  # Assume active if no expiration
        return 'present' in expiration_date.lower() or expiration_date == 'Never'
    
    def _extract_technologies(self, project_description: str) -> List[str]:
        """Extract technologies mentioned in project description"""
        tech_keywords = ['python', 'java', 'javascript', 'react', 'node', 'aws', 'docker', 'kubernetes']
        technologies = []
        
        desc_lower = project_description.lower()
        for tech in tech_keywords:
            if tech in desc_lower:
                technologies.append(tech.title())
        
        return technologies
    
    def _infer_project_domain(self, project_description: str) -> str:
        """Infer project domain from description"""
        desc_lower = project_description.lower()
        
        if any(word in desc_lower for word in ['web', 'website', 'frontend', 'backend']):
            return 'Web Development'
        elif any(word in desc_lower for word in ['machine learning', 'ai', 'data', 'analytics']):
            return 'Data Science/AI'
        elif any(word in desc_lower for word in ['mobile', 'app', 'ios', 'android']):
            return 'Mobile Development'
        else:
            return 'Software Development'
    
    def _classify_location_type(self, location: str) -> str:
        """Classify location type"""
        if ',' in location:
            return 'City, State/Country'
        else:
            return 'General Location'
    
    def _is_skill_relevant_to_job(self, skill: str, job_title: str, job_description: str) -> bool:
        """Check if skill is relevant to job"""
        text = f"{job_title} {job_description}".lower()
        return skill.lower() in text
    
    def _is_education_relevant_to_job(self, field_of_study: str, job_title: str) -> bool:
        """Check if education field is relevant to job"""
        relevance_map = {
            'computer science': ['engineer', 'developer', 'programmer', 'analyst'],
            'business': ['manager', 'analyst', 'consultant'],
            'engineering': ['engineer', 'technical', 'developer']
        }
        
        for field, job_keywords in relevance_map.items():
            if field in field_of_study:
                return any(keyword in job_title for keyword in job_keywords)
        
        return False
    
    def _are_related_cert_skill(self, cert_name: str, skill_name: str) -> bool:
        """Check if certification and skill are related"""
        related_pairs = {
            'aws': ['aws', 'cloud', 'amazon'],
            'microsoft': ['azure', 'microsoft', '.net'],
            'google': ['gcp', 'google', 'cloud'],
            'oracle': ['oracle', 'database', 'sql']
        }
        
        for cert_key, skill_keywords in related_pairs.items():
            if cert_key in cert_name:
                return any(keyword in skill_name for keyword in skill_keywords)
        
        return False
    
    def _calculate_data_quality_score(self, profile_data: Dict) -> float:
        """Calculate data quality score (0-1)"""
        score = 0.0
        max_score = 7.0
        
        # Basic info completeness
        if profile_data.get('name') and profile_data['name'] != 'Unknown':
            score += 1.0
        if profile_data.get('headline') and profile_data['headline'] != 'Not specified':
            score += 1.0
        
        # Experience data
        if profile_data.get('experience') and len(profile_data['experience']) > 0:
            score += 1.0
        
        # Education data
        if profile_data.get('education') and len(profile_data['education']) > 0:
            score += 1.0
        
        # Skills data
        if profile_data.get('skills') and len(profile_data['skills']) > 0:
            score += 1.0
        
        # Contact info
        if profile_data.get('contact_info') and len(profile_data['contact_info']) > 0:
            score += 1.0
        
        # Profile URL
        if profile_data.get('profile_url') and profile_data['profile_url'] != '':
            score += 1.0
        
        return score / max_score
    
    def export_to_json(self, kg: KnowledgeGraph, filename: str = None) -> str:
        """Export knowledge graph to JSON"""
        if not filename:
            filename = "linkedin_knowledge_graph.json"
        
        kg_dict = asdict(kg)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(kg_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge graph exported to {filename}")
        return filename
    
    def export_to_rdf(self, kg: KnowledgeGraph, filename: str = None) -> str:
        """Export knowledge graph to RDF/Turtle format"""
        if not filename:
            filename = "linkedin_knowledge_graph.ttl"
        
        rdf_content = "@prefix ln: <http://linkedin.com/ontology#> .\n"
        rdf_content += "@prefix foaf: <http://xmlns.com/foaf/0.1/> .\n"
        rdf_content += "@prefix org: <http://www.w3.org/ns/org#> .\n\n"
        
        # Add nodes
        for node in kg.nodes:
            rdf_content += f"ln:{node.id} a ln:{node.node_type} ;\n"
            rdf_content += f"  rdfs:label \"{node.label}\" ;\n"
            
            for prop, value in node.properties.items():
                if value and value != 'Not specified':
                    rdf_content += f"  ln:{prop} \"{value}\" ;\n"
            
            rdf_content = rdf_content.rstrip(';\n') + ' .\n\n'
        
        # Add edges
        for edge in kg.edges:
            rdf_content += f"ln:{edge.source} ln:{edge.relationship} ln:{edge.target} .\n"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(rdf_content)
        
        logger.info(f"RDF knowledge graph exported to {filename}")
        return filename
    
    def generate_llm_prompt(self, kg: KnowledgeGraph) -> str:
        """Generate an LLM-friendly prompt with the knowledge graph"""
        prompt = """# LinkedIn Profile Knowledge Graph

## Profile Overview
"""
        
        # Find person node
        person_node = next((n for n in kg.nodes if n.node_type == "Person"), None)
        if person_node:
            prompt += f"**Name:** {person_node.properties.get('name', 'Unknown')}\n"
            prompt += f"**Headline:** {person_node.properties.get('headline', 'Not specified')}\n"
            prompt += f"**Location:** {person_node.properties.get('location', 'Not specified')}\n\n"
        
        # Experience
        experience_nodes = [n for n in kg.nodes if n.node_type == "JobPosition"]
        if experience_nodes:
            prompt += "## Experience\n"
            for job in experience_nodes:
                company = ''
                # Find company node connected to this job
                for edge in kg.edges:
                    if edge.source == job.id and edge.relationship == "EMPLOYED_BY":
                        company_node = next((n for n in kg.nodes if n.id == edge.target), None)
                        if company_node:
                            company = company_node.label
                prompt += f"- **{job.label}** at {company if company else 'Unknown Company'} (Duration: {job.properties.get('duration', 'Not specified')})\n"
            prompt += "\n"

        # Education
        degree_nodes = [n for n in kg.nodes if n.node_type == "Degree"]
        if degree_nodes:
            prompt += "## Education\n"
            for degree in degree_nodes:
                # Find institution
                institution = ''
                for edge in kg.edges:
                    if edge.source == degree.id and edge.relationship == "AWARDED_BY":
                        inst_node = next((n for n in kg.nodes if n.id == edge.target), None)
                        if inst_node:
                            institution = inst_node.label
                prompt += f"- **{degree.label}** from {institution if institution else 'Unknown Institution'} (Duration: {degree.properties.get('duration', 'Not specified')})\n"
            prompt += "\n"

        # Skills
        skill_nodes = [n for n in kg.nodes if n.node_type == "Skill"]
        if skill_nodes:
            prompt += "## Skills\n"
            for skill in skill_nodes:
                prompt += f"- {skill.label}\n"
            prompt += "\n"

        # Certifications
        cert_nodes = [n for n in kg.nodes if n.node_type == "Certification"]
        if cert_nodes:
            prompt += "## Certifications\n"
            for cert in cert_nodes:
                org = cert.properties.get('issuing_organization', 'Unknown')
                prompt += f"- {cert.label} (Issued by: {org}, Status: {'Active' if cert.properties.get('is_active', True) else 'Expired'})\n"
            prompt += "\n"

        # Projects
        project_nodes = [n for n in kg.nodes if n.node_type == "Project"]
        if project_nodes:
            prompt += "## Projects\n"
            for project in project_nodes:
                prompt += f"- {project.label}\n"
            prompt += "\n"

        # Metadata
        prompt += "---\n"
        prompt += f"**Data Quality Score:** {kg.metadata.get('data_quality_score', 0):.2f}\n"
        prompt += f"**Profile URL:** {kg.metadata.get('profile_url', '')}\n"
        prompt += f"**Total Nodes:** {kg.metadata.get('total_nodes', 0)}\n"
        prompt += f"**Total Edges:** {kg.metadata.get('total_edges', 0)}\n"
        prompt += f"**Created At:** {kg.metadata.get('created_at', '')}\n"
        return prompt

    def visualize_graph(self, with_labels=True, figsize=(15, 10)):
        """Visualize the knowledge graph using matplotlib"""
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        plt.figure(figsize=figsize)
        node_colors = []
        for n in self.graph.nodes(data=True):
            t = n[1].get('node_type', 'Other')
            if t == 'Person':
                node_colors.append('skyblue')
            elif t == 'JobPosition':
                node_colors.append('orange')
            elif t == 'Company':
                node_colors.append('green')
            elif t == 'Skill':
                node_colors.append('purple')
            elif t == 'Degree':
                node_colors.append('red')
            elif t == 'EducationalInstitution':
                node_colors.append('brown')
            elif t == 'Certification':
                node_colors.append('gold')
            elif t == 'Project':
                node_colors.append('pink')
            else:
                node_colors.append('gray')
        nx.draw(self.graph, pos, with_labels=with_labels, node_color=node_colors, node_size=800, font_size=8, edge_color='gray', alpha=0.8)
        plt.title("LinkedIn Knowledge Graph")
        plt.show()

    def visualize_graph_plotly(self):
        """Visualize the knowledge graph using Plotly (interactive)"""
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        node_x = []
        node_y = []
        node_text = []
        for node in self.graph.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node[1].get('label', node[0])} ({node[1].get('node_type', '')})")
            node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=20,
                color=[hash(node[1].get('node_type', '')) % 10 for node in self.graph.nodes(data=True)],
                colorbar=dict(
                    thickness=15,
                    title='Node Type',  # <-- FIXED: removed 'titleside'
                    xanchor='left'
                ),
                line_width=2))
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='LinkedIn Knowledge Graph (Interactive)',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False)))
        fig.show()

class ProfessionalLinkedInScraper:
    def __init__(self, tavily_api_key: str):
        self.tavily = TavilyClient(api_key=tavily_api_key)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })

    def scrape_profile(self, profile_url: str) -> LinkedInProfile:
        """
        Main method to scrape LinkedIn profile with multiple fallback strategies
        """
        logger.info(f"Starting to scrape profile: {profile_url}")
        
        # Validate URL
        if not self._validate_linkedin_url(profile_url):
            raise ValueError("Invalid LinkedIn profile URL")
        
        # Try multiple extraction methods
        profile_data = None
        
        # Method 1: Advanced Tavily search with multiple queries
        try:
            profile_data = self._extract_with_advanced_tavily(profile_url)
            if profile_data and self._is_data_sufficient(profile_data):
                logger.info("Successfully extracted data using Advanced Tavily method")
                return profile_data
        except Exception as e:
            logger.warning(f"Advanced Tavily method failed: {e}")
        
        # Method 2: Direct content extraction
        try:
            profile_data = self._extract_with_direct_requests(profile_url)
            if profile_data and self._is_data_sufficient(profile_data):
                logger.info("Successfully extracted data using Direct requests method")
                return profile_data
        except Exception as e:
            logger.warning(f"Direct requests method failed: {e}")
        
        # Method 3: Google search for public LinkedIn data
        try:
            profile_data = self._extract_with_google_search(profile_url)
            if profile_data and self._is_data_sufficient(profile_data):
                logger.info("Successfully extracted data using Google search method")
                return profile_data
        except Exception as e:
            logger.warning(f"Google search method failed: {e}")
        
        # If all methods fail, return the best available data
        if profile_data:
            logger.warning("Returning partial data - not all extraction methods succeeded")
            return profile_data
        else:
            raise Exception("All extraction methods failed")

    def _validate_linkedin_url(self, url: str) -> bool:
        """Validate LinkedIn profile URL"""
        parsed = urlparse(url)
        return parsed.netloc.endswith('linkedin.com') and '/in/' in parsed.path

    def _extract_with_advanced_tavily(self, profile_url: str) -> LinkedInProfile:
        """Extract profile data using advanced Tavily search strategies"""
        username = self._extract_username_from_url(profile_url)
        
        # Multiple targeted search queries
        search_queries = [
            f'"{username}" site:linkedin.com/in experience work history',
            f'"{username}" site:linkedin.com/in education university degree',
            f'"{username}" site:linkedin.com/in skills certifications achievements',
            f'"{username}" site:linkedin.com/in projects publications',
            f'"{username}" LinkedIn profile contact information'
        ]
        
        all_results = []
        for query in search_queries:
            try:
                response = self.tavily.search(
                    query=query,
                    search_depth="advanced",
                    max_results=3,
                    include_raw_content=True,
                    include_images=True
                )
                all_results.extend(response.get('results', []))
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Search query failed: {query}, Error: {e}")
                continue
        
        return self._parse_comprehensive_data(all_results, profile_url)

    def _extract_with_direct_requests(self, profile_url: str) -> LinkedInProfile:
        """Attempt direct LinkedIn profile scraping"""
        try:
            response = self.session.get(profile_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                return self._extract_from_html(soup, profile_url)
            else:
                logger.warning(f"Direct request failed with status: {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Direct request method failed: {e}")
            return None

    def _extract_with_google_search(self, profile_url: str) -> LinkedInProfile:
        """Use general web search to find LinkedIn profile information"""
        username = self._extract_username_from_url(profile_url)
        
        # Search for cached or indexed versions of the profile
        search_queries = [
            f'cache:linkedin.com/in/{username}',
            f'"{username}" LinkedIn profile -site:linkedin.com',
            f'"{username}" professional experience education'
        ]
        
        all_results = []
        for query in search_queries:
            try:
                response = self.tavily.search(
                    query=query,
                    search_depth="basic",
                    max_results=5,
                    include_raw_content=True
                )
                all_results.extend(response.get('results', []))
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Google search failed: {query}, Error: {e}")
                continue
        
        return self._parse_comprehensive_data(all_results, profile_url)

    def _parse_comprehensive_data(self, search_results: List[Dict], profile_url: str) -> LinkedInProfile:
        """Parse search results into comprehensive profile data"""
        profile_data = {
            'name': 'Unknown',
            'headline': 'Not specified',
            'location': 'Not specified',
            'about': 'No summary available',
            'profile_url': profile_url,
            'profile_image_url': '',
            'background_image_url': '',
            'connections': 'Not specified',
            'followers': 'Not specified',
            'experience': [],
            'education': [],
            'skills': [],
            'certifications': [],
            'languages': [],
            'achievements': [],
            'projects': [],
            'publications': [],
            'contact_info': {}
        }
        
        for result in search_results:
            title = result.get('title', '')
            content = result.get('content', '')
            url = result.get('url', '')
            
            # Extract basic info from title
            self._extract_basic_info_from_title(title, profile_data)
            
            # Extract structured data from content
            self._extract_experience_data(content, profile_data)
            self._extract_education_data(content, profile_data)
            self._extract_skills_data(content, profile_data)
            self._extract_certifications_data(content, profile_data)
            self._extract_projects_data(content, profile_data)
            self._extract_contact_info(content, profile_data)
        
        # Clean and structure the data
        self._clean_and_validate_data(profile_data)
        
        return self._convert_to_dataclass(profile_data)

    def _extract_basic_info_from_title(self, title: str, profile_data: Dict):
        """Extract name, headline from search result titles"""
        if '|' in title and 'linkedin' in title.lower():
            parts = title.split('|')
            if len(parts) >= 2:
                if profile_data['name'] == 'Unknown':
                    profile_data['name'] = parts[0].strip()
                if profile_data['headline'] == 'Not specified':
                    profile_data['headline'] = parts[1].strip()

    def _extract_experience_data(self, content: str, profile_data: Dict):
        """Extract detailed work experience"""
        experience_patterns = [
            r'(?:Experience|Work History|Employment).*?(?=Education|Skills|$)',
            r'((?:Senior|Junior|Lead|Principal)?\s*(?:Software Engineer|Developer|Manager|Director|Analyst|Consultant|Designer))\s+at\s+([^,\n]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[-–]\s*([^,\n]+)\s*\(([^)]+)\)'
        ]
        
        for pattern in experience_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                exp_text = match.group(0) if match.groups() == () else ' '.join(match.groups())
                if len(exp_text) > 20 and len(exp_text) < 300:
                    experience_item = self._parse_experience_text(exp_text)
                    if experience_item and not self._is_duplicate_experience(experience_item, profile_data['experience']):
                        profile_data['experience'].append(experience_item)

    def _parse_experience_text(self, text: str) -> Dict:
        """Parse individual experience text into structured data"""
        # Enhanced parsing logic
        lines = text.split('\n')
        main_line = lines[0].strip()
        
        # Try to extract title and company
        if ' at ' in main_line:
            parts = main_line.split(' at ', 1)
            title = parts[0].strip()
            company = parts[1].strip()
        else:
            title = main_line
            company = 'Not specified'
        
        return {
            'title': title,
            'company': company,
            'duration': self._extract_duration(text),
            'location': self._extract_location(text),
            'description': text[:200] + '...' if len(text) > 200 else text,
            'employment_type': 'Not specified'
        }

    def _extract_education_data(self, content: str, profile_data: Dict):
        """Extract detailed education information"""
        education_patterns = [
            r'(?:Education|Academic Background).*?(?=Experience|Skills|$)',
            r'(Bachelor|Master|PhD|BS|MS|MBA|BA|MA)\s+(?:of|in|degree)?\s*([^,\n]+)\s*(?:from|at)?\s*([^,\n]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|College|Institute|School))[^,\n]*'
        ]
        
        for pattern in education_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                edu_text = match.group(0)
                if len(edu_text) > 10 and len(edu_text) < 200:
                    education_item = self._parse_education_text(edu_text)
                    if education_item and not self._is_duplicate_education(education_item, profile_data['education']):
                        profile_data['education'].append(education_item)

    def _parse_education_text(self, text: str) -> Dict:
        """Parse education text into structured data"""
        return {
            'institution': self._extract_institution(text),
            'degree': self._extract_degree(text),
            'field_of_study': self._extract_field_of_study(text),
            'duration': self._extract_duration(text),
            'grade': 'Not specified',
            'activities': 'Not specified'
        }

    def _extract_skills_data(self, content: str, profile_data: Dict):
        """Extract skills with better accuracy"""
        # Technical skills
        tech_skills = [
            'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust', 'Swift',
            'React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'Spring', 'Laravel',
            'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins', 'Git', 'SQL', 'NoSQL',
            'Machine Learning', 'AI', 'Data Science', 'Deep Learning', 'TensorFlow', 'PyTorch'
        ]
        
        # Soft skills
        soft_skills = [
            'Leadership', 'Communication', 'Project Management', 'Team Management',
            'Problem Solving', 'Critical Thinking', 'Analytical Skills', 'Creativity'
        ]
        
        all_skills = tech_skills + soft_skills
        
        for skill in all_skills:
            if skill.lower() in content.lower() and skill not in profile_data['skills']:
                profile_data['skills'].append(skill)

    def _extract_certifications_data(self, content: str, profile_data: Dict):
        """Extract certifications and licenses"""
        cert_patterns = [
            r'(?:Certified|Certificate|Certification|License)\s+([^,\n]+)',
            r'([A-Z]{2,})\s+(?:Certified|Certificate)',
            r'(?:AWS|Google|Microsoft|Oracle|Cisco|Adobe)\s+(?:Certified|Certificate)\s+([^,\n]+)'
        ]
        
        for pattern in cert_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                cert_text = match.group(0)
                if len(cert_text) > 5 and len(cert_text) < 100:
                    cert_item = {
                        'name': cert_text,
                        'issuing_organization': 'Not specified',
                        'issue_date': 'Not specified',
                        'expiration_date': 'Not specified',
                        'credential_id': 'Not specified',
                        'credential_url': 'Not specified'
                    }
                    if not self._is_duplicate_certification(cert_item, profile_data['certifications']):
                        profile_data['certifications'].append(cert_item)

    def _extract_projects_data(self, content: str, profile_data: Dict):
        """Extract projects and notable work"""
        project_keywords = ['project', 'developed', 'built', 'created', 'implemented', 'launched', 'designed']
        
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in project_keywords):
                if 20 < len(sentence) < 200:
                    if sentence not in profile_data['projects']:
                        profile_data['projects'].append(sentence)

    def _extract_contact_info(self, content: str, profile_data: Dict):
        """Extract contact information"""
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        if emails:
            profile_data['contact_info']['email'] = emails[0]
        
        # Phone pattern
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        phones = re.findall(phone_pattern, content)
        if phones:
            profile_data['contact_info']['phone'] = phones[0]
        
        # Website pattern
        website_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w.*))?)?'
        websites = re.findall(website_pattern, content)
        for website in websites:
            if 'linkedin.com' not in website:
                profile_data['contact_info']['website'] = website
                break

    def _extract_username_from_url(self, url: str) -> str:
        """Extract username from LinkedIn URL"""
        return url.split('/in/')[-1].strip('/')

    def _extract_duration(self, text: str) -> str:
        """Extract duration from text"""
        duration_patterns = [
            r'\b\d{4}\s*[-–]\s*\d{4}\b',
            r'\b\d{4}\s*[-–]\s*Present\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
            r'\b\d+\s+(?:year|month)s?\b'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return 'Not specified'

    def _extract_location(self, text: str) -> str:
        """Extract location from text"""
        location_patterns = [
            r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b',  # City, State
            r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b',  # City, Country
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return 'Not specified'

    def _extract_institution(self, text: str) -> str:
        """Extract educational institution"""
        institution_keywords = ['university', 'college', 'institute', 'school']
        words = text.split()
        
        for i, word in enumerate(words):
            if any(keyword in word.lower() for keyword in institution_keywords):
                # Try to get the full institution name
                start = max(0, i-3)
                end = min(len(words), i+2)
                return ' '.join(words[start:end])
        
        return 'Not specified'

    def _extract_degree(self, text: str) -> str:
        """Extract degree from education text"""
        degree_patterns = [
            r'\b(?:Bachelor|Master|PhD|BS|MS|MBA|BA|MA|BSc|MSc)\b',
            r'\b(?:B\.?A\.?|M\.?A\.?|B\.?S\.?|M\.?S\.?|Ph\.?D\.?|M\.?B\.?A\.?)\b'
        ]
        
        for pattern in degree_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return 'Not specified'

    def _extract_field_of_study(self, text: str) -> str:
        """Extract field of study"""
        common_fields = [
            'Computer Science', 'Engineering', 'Business', 'Marketing', 'Finance',
            'Psychology', 'Biology', 'Chemistry', 'Physics', 'Mathematics',
            'Economics', 'Political Science', 'History', 'English', 'Art'
        ]
        
        for field in common_fields:
            if field.lower() in text.lower():
                return field
        
        return 'Not specified'

    def _is_duplicate_experience(self, new_exp: Dict, existing_exps: List[Dict]) -> bool:
        """Check if experience is duplicate"""
        for exp in existing_exps:
            if (exp['title'].lower() == new_exp['title'].lower() and 
                exp['company'].lower() == new_exp['company'].lower()):
                return True
        return False

    def _is_duplicate_education(self, new_edu: Dict, existing_edus: List[Dict]) -> bool:
        """Check if education is duplicate"""
        for edu in existing_edus:
            if edu['institution'].lower() == new_edu['institution'].lower():
                return True
        return False

    def _is_duplicate_certification(self, new_cert: Dict, existing_certs: List[Dict]) -> bool:
        """Check if certification is duplicate"""
        for cert in existing_certs:
            if cert['name'].lower() == new_cert['name'].lower():
                return True
        return False

    def _clean_and_validate_data(self, profile_data: Dict):
        """Clean and validate extracted data"""
        # Limit list sizes
        profile_data['experience'] = profile_data['experience'][:10]
        profile_data['education'] = profile_data['education'][:5]
        profile_data['skills'] = profile_data['skills'][:20]
        profile_data['certifications'] = profile_data['certifications'][:10]
        profile_data['projects'] = profile_data['projects'][:10]
        
        # Clean text fields
        text_fields = ['name', 'headline', 'location', 'about']
        for field in text_fields:
            if isinstance(profile_data[field], str):
                profile_data[field] = profile_data[field].strip()

    def _convert_to_dataclass(self, profile_data: Dict) -> LinkedInProfile:
        """Convert dictionary to LinkedInProfile dataclass"""
        # Convert experience
        experiences = []
        for exp in profile_data['experience']:
            experiences.append(Experience(**exp))
        
        # Convert education
        educations = []
        for edu in profile_data['education']:
            educations.append(Education(**edu))
        
        # Convert certifications
        certifications = []
        for cert in profile_data['certifications']:
            certifications.append(Certification(**cert))
        
        return LinkedInProfile(
            name=profile_data['name'],
            headline=profile_data['headline'],
            location=profile_data['location'],
            about=profile_data['about'],
            profile_url=profile_data['profile_url'],
            profile_image_url=profile_data['profile_image_url'],
            background_image_url=profile_data['background_image_url'],
            connections=profile_data['connections'],
            followers=profile_data['followers'],
            experience=experiences,
            education=educations,
            skills=profile_data['skills'],
            certifications=certifications,
            languages=profile_data['languages'],
            achievements=profile_data['achievements'],
            projects=profile_data['projects'],
            publications=profile_data['publications'],
            contact_info=profile_data['contact_info']
        )

    def _extract_from_html(self, soup: BeautifulSoup, profile_url: str) -> LinkedInProfile:
        """Extract data directly from HTML (fallback method)"""
        # This is a simplified version - LinkedIn's HTML structure changes frequently
        profile_data = {
            'name': self._extract_name_from_html(soup),
            'headline': self._extract_headline_from_html(soup),
            'location': 'Not specified',
            'about': 'Not available',
            'profile_url': profile_url,
            'profile_image_url': '',
            'background_image_url': '',
            'connections': 'Not specified',
            'followers': 'Not specified',
            'experience': [],
            'education': [],
            'skills': [],
            'certifications': [],
            'languages': [],
            'achievements': [],
            'projects': [],
            'publications': [],
            'contact_info': {}
        }
        
        return self._convert_to_dataclass(profile_data)

    def _extract_name_from_html(self, soup: BeautifulSoup) -> str:
        """Extract name from HTML"""
        name_selectors = [
            'h1.text-heading-xlarge',
            'h1.top-card-layout__title',
            'h1[data-anonymize="person-name"]',
            '.pv-text-details__left-panel h1'
        ]
        
        for selector in name_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        
        return 'Unknown'

    def _extract_headline_from_html(self, soup: BeautifulSoup) -> str:
        """Extract headline from HTML"""
        headline_selectors = [
            '.text-body-medium.break-words',
            '.top-card-layout__headline',
            '.pv-text-details__left-panel .text-body-medium'
        ]
        
        for selector in headline_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        
        return 'Not specified'

    def _is_data_sufficient(self, profile: LinkedInProfile) -> bool:
        """Check if extracted data is sufficient"""
        if not profile:
            return False
        
        # Minimum requirements for sufficient data
        has_basic_info = (profile.name != 'Unknown' and 
                         profile.headline != 'Not specified')
        
        has_experience_or_education = (len(profile.experience) > 0 or 
                                     len(profile.education) > 0)
        
        return has_basic_info or has_experience_or_education

    def export_to_json(self, profile: LinkedInProfile, filename: str = None) -> str:
        """Export profile data to JSON"""
        if not filename:
            username = self._extract_username_from_url(profile.profile_url)
            filename = f"{username}_linkedin_profile.json"
        
        profile_dict = asdict(profile)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(profile_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Profile data exported to {filename}")
        return filename

    def get_profile_summary(self, profile: LinkedInProfile) -> str:
        """Generate a comprehensive profile summary"""
        summary = f"""
LinkedIn Profile Summary for {profile.name}
{'='*50}

Basic Information:
- Name: {profile.name}
- Headline: {profile.headline}
- Location: {profile.location}
- Profile URL: {profile.profile_url}

Professional Experience ({len(profile.experience)} positions):
"""
        
        for i, exp in enumerate(profile.experience[:5], 1):
            summary += f"{i}. {exp.title} at {exp.company} ({exp.duration})\n"
        
        summary += f"\nEducation ({len(profile.education)} institutions):\n"
        for i, edu in enumerate(profile.education[:3], 1):
            summary += f"{i}. {edu.degree} in {edu.field_of_study} from {edu.institution}\n"
        
        summary += f"\nSkills ({len(profile.skills)} skills):\n"
        summary += ", ".join(profile.skills[:10])
        
        if profile.certifications:
            summary += f"\n\nCertifications ({len(profile.certifications)}):\n"
            for i, cert in enumerate(profile.certifications[:5], 1):
                summary += f"{i}. {cert.name}\n"
        
        if profile.contact_info:
            summary += f"\nContact Information:\n"
            for key, value in profile.contact_info.items():
                summary += f"- {key.title()}: {value}\n"
        
        return summary

# --- Glue function to scrape and build knowledge graph ---
def scrape_and_build_kg(profile_url, tavily_api_key, visualize=None):
    """Scrape LinkedIn profile and build knowledge graph in one go. Optionally visualize."""
    scraper = ProfessionalLinkedInScraper(tavily_api_key)
    profile = scraper.scrape_profile(profile_url)
    json_file = scraper.export_to_json(profile)
    with open(json_file, "r", encoding="utf-8") as f:
        profile_data = json.load(f)
    kg_builder = LinkedInKnowledgeGraphBuilder()
    kg = kg_builder.build_knowledge_graph(profile_data)
    kg_builder.export_to_json(kg)
    kg_builder.export_to_rdf(kg)
    print(kg_builder.generate_llm_prompt(kg))
    if visualize == 'matplotlib':
        kg_builder.visualize_graph()
    elif visualize == 'plotly':
        kg_builder.visualize_graph_plotly()

# --- CLI main function ---
def main():
    print("Professional LinkedIn Profile Scraper & Knowledge Graph Builder")
    print("="*60)
    tavily_api_key = input("Enter your Tavily API key: ").strip()
    profile_url = input("Enter LinkedIn profile URL: ").strip()
    visualize = None
    vis_choice = input("Do you want to visualize the knowledge graph? (none/matplotlib/plotly): ").strip().lower()
    if vis_choice in ('matplotlib', 'plotly'):
        visualize = vis_choice
    try:
        print("\nExtracting profile data and building knowledge graph... This may take a moment.")
        scrape_and_build_kg(profile_url, tavily_api_key, visualize=visualize)
    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"Scraping or KG build failed: {e}")

if __name__ == "__main__":
    main()