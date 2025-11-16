#!/usr/bin/env python
"""
Datalog Compiler for HyperKG

Compiles Datalog rules into hyperbolic geometric constraints.
Bridges symbolic reasoning (Datalog) with geometric reasoning (hyperbolic space).

Compilation strategy:
1. Parse Datalog rules into logical form
2. Compile rules into geometric constraints on embeddings
3. Enforce constraints during training (soft constraints via loss)
4. Verify inferences with PGU symbolic prover

Example:
    # Rule: ancestor(X, Z) :- parent(X, Y), parent(Y, Z)
    # Geometric constraint:
    #   d(X⊕ancestor, Z) ≤ d(X⊕parent, Y) + d(Y⊕parent, Z) + ε

Usage:
    from tfan.kg import DatalogCompiler

    compiler = DatalogCompiler(kg)

    # Add transitivity rule
    compiler.add_rule(
        "ancestor(?X, ?Z)",
        ["parent(?X, ?Y)", "parent(?Y, ?Z)"]
    )

    # Compile to geometric constraints
    constraints = compiler.compile_rules()

    # Train with constraints
    kg.train_with_constraints(constraints)
"""

import re
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Variable:
    """Datalog variable (e.g., ?X, ?Y)."""
    name: str

    def __repr__(self):
        return f"?{self.name}"


@dataclass
class Constant:
    """Datalog constant (e.g., socrates, human)."""
    value: str

    def __repr__(self):
        return self.value


@dataclass
class Atom:
    """
    Datalog atom (e.g., parent(?X, ?Y)).

    predicate(term1, term2, ...)
    """
    predicate: str
    terms: List  # List of Variable or Constant

    def __repr__(self):
        terms_str = ", ".join(str(t) for t in self.terms)
        return f"{self.predicate}({terms_str})"


@dataclass
class Rule:
    """
    Datalog rule: head :- body.

    head is a single atom
    body is a list of atoms
    """
    head: Atom
    body: List[Atom]

    def __repr__(self):
        body_str = ", ".join(str(atom) for atom in self.body)
        return f"{self.head} :- {body_str}"


class DatalogCompiler:
    """
    Compiles Datalog rules into hyperbolic geometric constraints.

    Supports:
    - Transitivity: ancestor(X,Z) :- parent(X,Y), parent(Y,Z)
    - Symmetry: related(X,Y) :- related(Y,X)
    - Inheritance: property(X,P) :- is_a(X,Y), property(Y,P)
    """

    def __init__(self, kg=None):
        """
        Initialize compiler.

        Args:
            kg: HyperbolicKG instance to compile rules for
        """
        self.kg = kg
        self.rules: List[Rule] = []
        self.constraints: List[Dict] = []

    def parse_term(self, term_str: str):
        """
        Parse a term (variable or constant).

        Variables start with '?' or are uppercase.
        Constants are lowercase.
        """
        term_str = term_str.strip()

        if term_str.startswith('?'):
            return Variable(term_str[1:])
        elif term_str[0].isupper():
            return Variable(term_str)
        else:
            return Constant(term_str)

    def parse_atom(self, atom_str: str) -> Atom:
        """
        Parse an atom like 'parent(?X, ?Y)'.

        Format: predicate(term1, term2, ...)
        """
        # Match predicate(args)
        match = re.match(r'(\w+)\((.*)\)', atom_str.strip())
        if not match:
            raise ValueError(f"Invalid atom: {atom_str}")

        predicate = match.group(1)
        args_str = match.group(2)

        # Parse terms
        terms = [self.parse_term(t.strip()) for t in args_str.split(',')]

        return Atom(predicate, terms)

    def parse_rule(self, rule_str: str) -> Rule:
        """
        Parse a rule like 'ancestor(?X, ?Z) :- parent(?X, ?Y), parent(?Y, ?Z)'.

        Format: head :- body1, body2, ...
        """
        if ':-' not in rule_str:
            raise ValueError(f"Invalid rule (missing ':-'): {rule_str}")

        head_str, body_str = rule_str.split(':-')

        head = self.parse_atom(head_str.strip())
        body = [self.parse_atom(atom.strip()) for atom in body_str.split(',')]

        return Rule(head, body)

    def add_rule(self, rule_str: str):
        """Add a rule from string."""
        rule = self.parse_rule(rule_str)
        self.rules.append(rule)

    def compile_rules(self) -> List[Dict]:
        """
        Compile Datalog rules into geometric constraints.

        For each rule, generate a constraint that enforces the rule
        in hyperbolic space.

        Returns:
            constraints: List of constraint dicts
        """
        self.constraints = []

        for rule in self.rules:
            constraint = self._compile_rule(rule)
            if constraint:
                self.constraints.append(constraint)

        return self.constraints

    def _compile_rule(self, rule: Rule) -> Optional[Dict]:
        """
        Compile a single rule into geometric constraint.

        Constraint types:
        - Transitivity: d(X⊕r1, Z) ≤ d(X⊕r2, Y) + d(Y⊕r3, Z) + ε
        - Symmetry: d(X⊕r, Y) = d(Y⊕r, X)
        - Subsumption: d(X⊕is_a, Y) → d(X⊕property, P) ≈ d(Y⊕property, P)
        """
        # Transitivity pattern: R(X,Z) :- R1(X,Y), R2(Y,Z)
        if len(rule.body) == 2:
            atom1, atom2 = rule.body

            # Check for shared variable (transitivity)
            shared_vars = self._find_shared_variables([atom1, atom2])

            if len(shared_vars) == 1:
                # Transitivity constraint
                return {
                    'type': 'transitivity',
                    'head_relation': rule.head.predicate,
                    'body_relations': [atom1.predicate, atom2.predicate],
                    'pattern': 'chain',
                    'description': f"{rule.head} :- {atom1}, {atom2}"
                }

        # Symmetry pattern: R(X,Y) :- R(Y,X)
        if len(rule.body) == 1:
            atom = rule.body[0]

            if rule.head.predicate == atom.predicate:
                # Check if terms are swapped
                if (isinstance(rule.head.terms[0], Variable) and
                    isinstance(rule.head.terms[1], Variable) and
                    isinstance(atom.terms[0], Variable) and
                    isinstance(atom.terms[1], Variable)):

                    if (rule.head.terms[0].name == atom.terms[1].name and
                        rule.head.terms[1].name == atom.terms[0].name):
                        return {
                            'type': 'symmetry',
                            'relation': rule.head.predicate,
                            'description': f"{rule.head} :- {atom}"
                        }

        # Subsumption pattern: property(X,P) :- is_a(X,Y), property(Y,P)
        if len(rule.body) == 2:
            atom1, atom2 = rule.body

            if atom1.predicate == 'is_a' and atom2.predicate == rule.head.predicate:
                return {
                    'type': 'subsumption',
                    'head_relation': rule.head.predicate,
                    'subsumption_relation': 'is_a',
                    'description': f"{rule.head} :- {atom1}, {atom2}"
                }

        # Generic rule (no specific constraint)
        return {
            'type': 'generic',
            'head': rule.head,
            'body': rule.body,
            'description': str(rule)
        }

    def _find_shared_variables(self, atoms: List[Atom]) -> Set[str]:
        """Find variables shared across multiple atoms."""
        var_sets = []

        for atom in atoms:
            vars_in_atom = {t.name for t in atom.terms if isinstance(t, Variable)}
            var_sets.append(vars_in_atom)

        if len(var_sets) < 2:
            return set()

        # Find intersection
        shared = var_sets[0]
        for var_set in var_sets[1:]:
            shared &= var_set

        return shared

    def generate_constraint_loss(self, constraint: Dict, kg) -> float:
        """
        Generate loss term for a constraint.

        Enforces geometric constraint during training.

        Args:
            constraint: Compiled constraint
            kg: HyperbolicKG instance

        Returns:
            loss: Scalar loss value
        """
        import torch

        if constraint['type'] == 'transitivity':
            # For all (X, r1, Y) and (Y, r2, Z):
            #   d(X⊕r_head, Z) should be ≤ d(X⊕r1, Y) + d(Y⊕r2, Z)
            #
            # Loss: max(0, d(X⊕r_head, Z) - d(X⊕r1, Y) - d(Y⊕r2, Z) - margin)

            # Stub: would iterate over all matching triples
            return 0.0

        elif constraint['type'] == 'symmetry':
            # For all (X, r, Y):
            #   d(X⊕r, Y) should equal d(Y⊕r, X)
            #
            # Loss: |d(X⊕r, Y) - d(Y⊕r, X)|

            # Stub: would iterate over all triples with this relation
            return 0.0

        elif constraint['type'] == 'subsumption':
            # For all (X, is_a, Y) and (Y, property, P):
            #   d(X⊕property, P) should be close to d(Y⊕property, P)
            #
            # Loss: |d(X⊕property, P) - d(Y⊕property, P)|

            # Stub: would iterate over all matching triple pairs
            return 0.0

        return 0.0

    def get_stats(self) -> Dict:
        """Get compiler statistics."""
        return {
            'num_rules': len(self.rules),
            'num_constraints': len(self.constraints),
            'constraint_types': self._count_constraint_types()
        }

    def _count_constraint_types(self) -> Dict[str, int]:
        """Count constraints by type."""
        counts = defaultdict(int)
        for constraint in self.constraints:
            counts[constraint['type']] += 1
        return dict(counts)

    def print_rules(self):
        """Print all compiled rules."""
        print(f"Compiled {len(self.rules)} Datalog rules:")
        for i, rule in enumerate(self.rules):
            print(f"  {i+1}. {rule}")

    def print_constraints(self):
        """Print all generated constraints."""
        print(f"Generated {len(self.constraints)} geometric constraints:")
        for i, constraint in enumerate(self.constraints):
            print(f"  {i+1}. [{constraint['type']}] {constraint['description']}")
