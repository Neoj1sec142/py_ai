'''
The Gale-Shapley algorithm, also known as the Stable Marriage Problem, is a well-known algorithm used to find a stable matching between two equally sized sets of elements given an ordered preference for each element. A matching is considered "stable" if there are no two elements that would prefer each other over their current partners.


Initialize all men and women as free.
While there is a free man who still has a woman to propose to:
The man proposes to the most preferred woman to whom he has not yet proposed.
If the woman is free, she accepts the proposal (they become engaged).
If the woman is not free:
    She checks if she prefers this new proposal over her current engagement.
    If she prefers the new proposal, she replaces her current engagement with the new one, making her previous fiancé free again.
    Otherwise, she rejects the new proposal, and the man remains free.
'''


def stable_marriage(men_preferences, women_preferences):
    # Initialize all men and women as free
    free_men = list(men_preferences.keys())
    engaged = {}
    proposed = {m: [] for m in men_preferences}

    while free_men:
        man = free_men[0]
        man_prefs = men_preferences[man]
        
        # Find the first woman to propose to whom he hasn't proposed yet
        for woman in man_prefs:
            if woman not in proposed[man]:
                proposed[man].append(woman)
                break
        
        # If the woman is free, engage them
        if woman not in engaged:
            engaged[woman] = man
            free_men.remove(man)
        else:
            current_partner = engaged[woman]

            # Check if the woman prefers this new man
            if women_preferences[woman].index(man) < women_preferences[woman].index(current_partner):
                # She prefers the new man, so engage to the new man and free the old partner
                engaged[woman] = man
                free_men.remove(man)
                free_men.append(current_partner)
    
    # Invert the dictionary to get the final pairing
    return {v: k for k, v in engaged.items()}

# Example usage
men_preferences = {
    "A": ["X", "Y", "Z"],
    "B": ["Y", "X", "Z"],
    "C": ["Z", "Y", "X"]
}

women_preferences = {
    "X": ["B", "A", "C"],
    "Y": ["C", "B", "A"],
    "Z": ["A", "C", "B"]
}

print(stable_marriage(men_preferences, women_preferences))
