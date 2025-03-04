import sys
import re
import pandas as pd
#from numpy.core.numeric import NaN

# Read in the file
file = pd.read_csv(sys.argv[1], sep='\t')

# The token to be searched in the document
target_token = sys.argv[2]
# The register the token is concerned with
target_register = sys.argv[3]

def preprocess_token(token):
    """
    Applies the same preprocessing rules to a token (lowercasing, regex
    filtering) as used with the original keyword search, so that using extracted
    keywords as search queries will work identically.
    """
    return re.sub(r"[^-\w\s'’/<>]", "", str(token)).lower()

# Filter documents based on preprocessed tokens, but join original tokens
filtered_docs = file.groupby('id').filter(
	# Whether target token is in the document
    lambda group: target_token in group['token'].apply(preprocess_token).values
    # Whether there is an actual prediction at all
    and not group.pred.isna().any()
    # Whether the prediction is an exact true positive
    and set(eval(group.label.unique()[0])) == set(eval(group.pred.unique()[0]))
	# And whether the label is the one the keyword is concerned with
    and target_register in group.label.unique()[0]
).groupby('id')['token'].apply(
    lambda tokens: ' '.join(tokens.fillna("").astype(str))
)

# Save to CSV
filtered_docs.to_csv(f"result-documents/{target_register}-{target_token}.csv")
