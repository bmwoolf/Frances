"""
Flux Simulator - Simulated Lab for FBA predictions
Downloads BiGG models and runs flux balance analysis simulations
"""

import cobra
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FluxSimulator:
    """Simulated lab for flux balance analysis"""
    
    def __init__(self):
        self.models = {}  # Cache downloaded models
        self.available_models = {
            'E.coli': 'iJO1366',  # E. coli core model
            'S.cerevisiae': 'iMM904',  # Yeast model
        }
    
    def get_model(self, host: str) -> cobra.Model:
        """Download and load BiGG model for host organism"""
        if host in self.models:
            return self.models[host]
        
        try:
            model_id = self.available_models.get(host, 'iJO1366')  # Default to E. coli
            logger.info(f"Downloading BiGG model: {model_id}")
            # Try different methods to load BiGG models
            try:
                model = cobra.io.load_bigg_model(model_id)
            except AttributeError:
                # Try alternative method
                model = cobra.io.read_sbml_model(f"https://bigg.ucsd.edu/static/models/{model_id}.xml")
            
            self.models[host] = model
            return model
        except Exception as e:
            logger.warning(f"Could not download {model_id}, using toy model: {e}")
            return self._create_toy_model(host)
    
    def _create_toy_model(self, host: str) -> cobra.Model:
        """Create a simple toy model for testing when BiGG models aren't available"""
        model = cobra.Model(f"toy_{host}")
        
        # Add metabolites
        glucose = cobra.Metabolite("glc", name="Glucose")
        biomass = cobra.Metabolite("biomass", name="Biomass")
        target = cobra.Metabolite("target", name="Target Product")
        
        # Add reactions
        uptake = cobra.Reaction("EX_glc")
        uptake.add_metabolites({glucose: -1})
        uptake.lower_bound = -10
        uptake.upper_bound = -1  # Allow glucose uptake (negative means uptake)
        
        growth = cobra.Reaction("BIOMASS")
        growth.add_metabolites({glucose: -0.5, biomass: 1})
        growth.lower_bound = 0.1  # Minimum growth requirement
        
        production = cobra.Reaction("PRODUCTION")
        production.add_metabolites({glucose: -0.3, target: 1})
        production.lower_bound = 0
        
        # Add exchange reactions
        ex_target = cobra.Reaction("EX_target")
        ex_target.add_metabolites({target: -1})
        ex_target.lower_bound = 0
        
        model.add_reactions([uptake, growth, production, ex_target])
        model.objective = ex_target  # Maximize target production
        
        return model
    
    def simulate_knockout(self, host: str, knockout_genes: List[str]) -> Dict:
        """Simulate gene knockouts and return predicted yield"""
        model = self.get_model(host)
        
        try:
            # Create a copy for simulation
            sim_model = model.copy()
            
            # Apply knockouts
            for gene_id in knockout_genes:
                if gene_id in sim_model.genes:
                    sim_model.genes.get_by_id(gene_id).knock_out()
            
            # Run FBA
            solution = sim_model.optimize()
            
            if solution.status == 'optimal':
                # Extract target production (simplified)
                target_flux = self._extract_target_flux(sim_model, solution)
                biomass_flux = solution.objective_value
                
                return {
                    "status": "optimal",
                    "predicted_yield": float(target_flux),
                    "biomass_flux": float(biomass_flux),
                    "knockout_genes": knockout_genes
                }
            else:
                return {
                    "status": "infeasible",
                    "predicted_yield": 0.0,
                    "biomass_flux": 0.0,
                    "knockout_genes": knockout_genes
                }
                
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return {
                "status": "error",
                "predicted_yield": 0.0,
                "biomass_flux": 0.0,
                "knockout_genes": knockout_genes,
                "error": str(e)
            }
    
    def simulate_overexpression(self, host: str, overexpress_genes: List[str]) -> Dict:
        """Simulate gene overexpression and return predicted yield"""
        model = self.get_model(host)
        
        try:
            sim_model = model.copy()
            
            # Apply overexpression (increase flux bounds)
            for gene_id in overexpress_genes:
                if gene_id in sim_model.genes:
                    # Increase reaction bounds for reactions catalyzed by this gene
                    gene = sim_model.genes.get_by_id(gene_id)
                    for reaction in gene.reactions:
                        reaction.upper_bound *= 2.0  # Double the flux
            
            # Run FBA
            solution = sim_model.optimize()
            
            if solution.status == 'optimal':
                target_flux = self._extract_target_flux(sim_model, solution)
                biomass_flux = solution.objective_value
                
                return {
                    "status": "optimal",
                    "predicted_yield": float(target_flux),
                    "biomass_flux": float(biomass_flux),
                    "overexpress_genes": overexpress_genes
                }
            else:
                return {
                    "status": "infeasible",
                    "predicted_yield": 0.0,
                    "biomass_flux": 0.0,
                    "overexpress_genes": overexpress_genes
                }
                
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return {
                "status": "error",
                "predicted_yield": 0.0,
                "biomass_flux": 0.0,
                "overexpress_genes": overexpress_genes,
                "error": str(e)
            }
    
    def _extract_target_flux(self, model: cobra.Model, solution) -> float:
        """Extract target product flux from solution"""
        # Look for common target product reactions
        target_reactions = [
            'EX_target', 'PRODUCTION', 'EX_limonene', 'EX_ethanol',
            'EX_acetate', 'EX_lactate', 'EX_succinate'
        ]
        
        for rxn_id in target_reactions:
            if rxn_id in model.reactions:
                return abs(solution.fluxes.get(rxn_id, 0.0))
        
        # If no specific target found, return a proxy based on biomass
        return solution.objective_value * 0.1  # 10% of biomass as proxy
    
    def run_full_simulation(self, host: str, knockouts: List[str], 
                          overexpressions: List[str]) -> Dict:
        """Run complete simulation with both knockouts and overexpressions"""
        model = self.get_model(host)
        
        try:
            sim_model = model.copy()
            
            # Apply knockouts
            for gene_id in knockouts:
                if gene_id in sim_model.genes:
                    sim_model.genes.get_by_id(gene_id).knock_out()
            
            # Apply overexpressions
            for gene_id in overexpressions:
                if gene_id in sim_model.genes:
                    gene = sim_model.genes.get_by_id(gene_id)
                    for reaction in gene.reactions:
                        reaction.upper_bound *= 2.0
            
            # Run FBA
            solution = sim_model.optimize()
            
            if solution.status == 'optimal':
                target_flux = self._extract_target_flux(sim_model, solution)
                biomass_flux = solution.objective_value
                
                return {
                    "status": "optimal",
                    "predicted_yield": float(target_flux),
                    "biomass_flux": float(biomass_flux),
                    "knockout_genes": knockouts,
                    "overexpress_genes": overexpressions
                }
            else:
                return {
                    "status": "infeasible",
                    "predicted_yield": 0.0,
                    "biomass_flux": 0.0,
                    "knockout_genes": knockouts,
                    "overexpress_genes": overexpressions
                }
                
        except Exception as e:
            logger.error(f"Full simulation error: {e}")
            return {
                "status": "error",
                "predicted_yield": 0.0,
                "biomass_flux": 0.0,
                "knockout_genes": knockouts,
                "overexpress_genes": overexpressions,
                "error": str(e)
            }

# Global simulator instance
simulator = FluxSimulator() 