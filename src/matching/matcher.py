from typing import List
from config import settings
from src.models import Action, SearchResult, MatchResult


class IntentMatcher:
    
    def __init__(self):
        self.no_match_threshold = settings.NO_MATCH_THRESHOLD
        self.confusion_margin = settings.CONFUSION_MARGIN
    
    def match(
        self, 
        search_result: SearchResult, 
        all_results: List[dict]
    ) -> MatchResult:
        # Step 1: Check if we have any results
        if not all_results:
            return MatchResult(
                match_type="none",
                actions=[],
                reasoning="No search results returned",
                search_results=search_result
            )
        
        # Extract the best (lowest) distance
        best_distance = all_results[0]["distance"]
        
        # Step 2: Check for NO MATCH (distance too large)
        if best_distance > self.no_match_threshold:
            return MatchResult(
                match_type="none",
                actions=[],
                reasoning=(
                    f"Best match distance ({best_distance:.4f}) exceeds "
                    f"threshold ({self.no_match_threshold}). "
                    f"User message doesn't match any available actions closely enough."
                ),
                search_results=search_result
            )
        
        # Step 3: Check for CONFUSION (top results too close)
        if len(all_results) >= 2:
            # Get the second-best distance
            second_distance = all_results[1]["distance"]
            
            # Calculate the difference between best and second-best
            distance_diff = second_distance - best_distance
            
            # If the difference is small, it's confusing which one to pick
            if distance_diff < self.confusion_margin:
                # Return the top 3 actions (or fewer if less than 3 results)
                confused_actions = [
                    result["action"] 
                    for result in all_results[:3]
                ]
                
                return MatchResult(
                    match_type="multiple",
                    actions=confused_actions,
                    reasoning=(
                        f"Top results are too close: "
                        f"1st={best_distance:.4f}, 2nd={second_distance:.4f} "
                        f"(diff={distance_diff:.4f}, margin={self.confusion_margin}). "
                        f"Multiple actions seem equally relevant."
                    ),
                    search_results=search_result
                )
        
        # Step 4: Clear winner!
        return MatchResult(
            match_type="single",
            actions=[all_results[0]["action"]],
            reasoning=(
                f"Clear match found with distance {best_distance:.4f} "
                f"(threshold: {self.no_match_threshold})"
            ),
            search_results=search_result
        )
    
    def format_output(
        self, 
        match_result: MatchResult, 
        all_results: List[dict]
    ) -> str:
        # Build the output line by line
        lines = []
        
        # Header
        lines.append("\n" + "=" * 80)
        lines.append("SEARCH RESULTS")
        lines.append("=" * 80)
        
        # Search metadata
        if match_result.search_results:
            sr = match_result.search_results
            lines.append(f"\n⏱️  Search Method: {sr.search_method}")
            lines.append(f"⏱️  Search Time: {sr.time_taken:.4f}s")
        
        # Top matches
        lines.append(f"\n🎯 Top Matches:")
        for i, result in enumerate(all_results[:3], 1):
            action = result["action"]
            distance = result["distance"]
            lines.append(f"\n  {i}. {action.title}")
            lines.append(f"     Distance: {distance:.4f}")
            lines.append(f"     Description: {action.description}")
        
        # Match result
        lines.append("\n" + "-" * 80)
        lines.append("MATCH RESULT")
        lines.append("-" * 80)
        
        # Display match type with appropriate emoji
        match_type_display = {
            "none": "❌ NO MATCH",
            "single": "✅ SINGLE MATCH",
            "multiple": "🤔 MULTIPLE MATCHES (Confusion)"
        }
        lines.append(f"\n📊 Match Type: {match_type_display.get(match_result.match_type, match_result.match_type)}")
        
        # Display recommended actions
        if match_result.actions:
            lines.append(f"\n🎯 Recommended Action(s):")
            for action in match_result.actions:
                lines.append(f"   • {action.title}")
        else:
            lines.append(f"\n🎯 Recommended Action(s): None")
        
        # Display reasoning
        lines.append(f"\n💡 Reasoning:")
        lines.append(f"   {match_result.reasoning}")
        
        # Footer
        lines.append("\n" + "=" * 80 + "\n")
        
        return "\n".join(lines)

