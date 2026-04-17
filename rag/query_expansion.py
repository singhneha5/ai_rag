import re


class QueryExpander:
    def __init__(self):
        # Simple synonym map for financial terms
        self.synonyms = {
            "revenue": ["sales", "income", "earnings"],
            "profit": ["net income", "earnings", "bottom line"],
            "loss": ["deficit", "negative earnings"],
            "growth": ["increase", "expansion", "rise"],
            "decline": ["decrease", "fall", "drop"],
            "assets": ["resources", "holdings", "property"],
            "liabilities": ["debts", "obligations", "amounts owed"],
            "cash flow": ["liquidity", "cash movement"],
            "roi": ["return on investment", "returns"],
            "margin": ["profitability ratio"],
            "expense": ["cost", "expenditure"],
            "investment": ["capital", "funds"],
        }

    def expand_query(self, query):
        """
        Expand query with synonyms for better retrieval
        Returns list of expanded queries
        """
        expanded = [query]  # Original query

        # Find synonyms in the query
        for term, synonyms in self.synonyms.items():
            if term.lower() in query.lower():
                for synonym in synonyms:
                    expanded_query = re.sub(
                        r"\b" + term + r"\b", f"({term} {synonym})", query, flags=re.IGNORECASE
                    )
                    expanded.append(expanded_query)
                break  # Only expand first matched term

        return expanded[:3]  # Return top 3 variations
