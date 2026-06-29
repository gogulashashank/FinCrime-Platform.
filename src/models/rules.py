import yaml
import polars as pl
import pandas as pd
import json

class RuleEngine:
    def __init__(self, config_path='configs/rules.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.rules = self.config.get('rules', [])
        
    def evaluate(self, base_table: pl.DataFrame) -> pl.DataFrame:
        print(f"Evaluating {len(self.rules)} rules on {len(base_table)} transactions...")
        
        # Add columns for rule hits and scores
        # We will use pandas for ease of row-level dynamic rule evaluation, 
        # or do it in polars via expressions. Polars expressions are much faster.
        
        # We will build a list of exprs for each rule
        rule_exprs = []
        rule_scores = []
        
        for rule in self.rules:
            rule_id = rule['id']
            score = rule['score']
            conditions = rule['conditions']
            
            # Combine conditions with AND
            combined_expr = pl.lit(True)
            for cond in conditions:
                feat = cond['feature']
                op = cond['operator']
                val = cond['value']
                
                if op == '>=':
                    expr = pl.col(feat) >= val
                elif op == '<=':
                    expr = pl.col(feat) <= val
                elif op == '==':
                    expr = pl.col(feat) == val
                elif op == '>':
                    expr = pl.col(feat) > val
                elif op == '<':
                    expr = pl.col(feat) < val
                else:
                    expr = pl.lit(False)
                    
                combined_expr = combined_expr & expr
                
            rule_exprs.append(combined_expr.alias(f"hit_{rule_id}"))
            
        # Apply all rule expressions
        scored_table = base_table.with_columns(rule_exprs)
        
        # Calculate total risk score and list of triggered rules
        # Convert to Pandas for complex row-wise aggregation (easier to read/explain)
        df = scored_table.to_pandas()
        
        df['risk_score'] = 0
        df['triggered_rules'] = [[] for _ in range(len(df))]
        
        for rule in self.rules:
            rule_id = rule['id']
            score = rule['score']
            hit_col = f"hit_{rule_id}"
            
            mask = df[hit_col] == True
            df.loc[mask, 'risk_score'] += score
            
            # Append rule id to list
            df.loc[mask, 'triggered_rules'] = df.loc[mask, 'triggered_rules'].apply(lambda x: x + [rule_id])
            
        # Cap score at 100
        df['risk_score'] = df['risk_score'].clip(upper=100)
        
        # Filter only alerts (risk_score > 0)
        alerts = df[df['risk_score'] > 0].copy()
        alerts['triggered_rules'] = alerts['triggered_rules'].apply(lambda x: json.dumps(x))
        
        print(f"Generated {len(alerts)} alerts from rule engine.")
        return pl.from_pandas(alerts)
