# Gender Bias in Children Fairy Tales
**Contributors:** Ran An, Li Feng, Selina Wang

## Introduction

Children's fairy tales have been part of our cultural heritage for centuries, often shaping children's perceptions of societal norms and values. However, these stories frequently contain gender stereotypes and biases. Our project aims to quantitatively uncover these biases using deep learning techniques. We trained Recurrent Neural Networks (RNNs), Bi-directional Long Short-Term Memory (LSTM) networks, and Transformer models on fairy tale texts and analyzed the resultant word embeddings to detect hidden gender biases.

## Related Work

Schroder et al. outline various methods to define and measure bias, including geometrical bias definitions, classification, clustering bias, and cosine-based measures like WEAT, MAC, and direct bias. Kurita et al. use the masked language method in BERT to compute associations between gendered words and attributes, validating their approach through a case study on gender bias in Gender Pronoun Resolution.

## Methodology

### Data Collection
We used 48 fairy tale books from Project Gutenberg, translated into English.

### Text Preprocessing
- Tokenization
- Removing stop words and punctuation
- Lemmatization
- Setting uncommon words to `<UNK>`

### Model Training
We trained three model architectures:
1. Basic RNN for predicting the next word in a sequence
2. Bi-directional LSTM for a Masked Language Modeling (MLM) task
3. Transformer for a Masked Language Modeling (MLM) task

### MLM Model Architecture
For the MLM task, we randomly masked one word in each input sequence and aimed to predict it. The bi-directional LSTM and Transformer blocks used context from both before and after the masked word.

### Bias Measurement
1. **Masked Language Modeling Task**: We examined predicted probabilities for gender associations with words (e.g., 'flower' with 'she' vs. 'he').
2. **Embedding Matrices**: We analyzed the cosine similarity between word embeddings to detect gender bias.
3. **Q Value Calculation**: We introduced a new value, Q, based on the eigenvalues of the covariance matrix to measure bias in word pairs.

## Results

### Masked Language Modeling
Both the Transformer and bi-LSTM models revealed biases, though inconsistencies were noted due to the transformer's higher loss.

| Input Sentence          | Bi-LSTM ('she') | Bi-LSTM ('he') | Transformer ('she') | Transformer ('he') |
|-------------------------|-----------------|----------------|----------------------|--------------------|
| [mask] is brave         | 0.284           | 5.286          | 1.292                | 4.378              |
| [mask] go to adventure  | 2.754           | 6.394          | 0.031                | 12.821             |
| [mask] is dancer        | 3.714           | 3.499          | 0.750                | 3.807              |
| [mask] is powerful      | 0.479           | 1.038          | 4.393                | 9.716              |
| [mask] like flower      | 5.001           | 0.439          | 0.038                | 0.649              |
| [mask] is evil          | 10.290          | 4.771          | 15.690               | 5.514              |
| [mask] is farmer        | 0.554           | 0.335          | 0.103                | 11.242             |
| [mask] is doctor        | 1.685           | 0.042          | 0.471                | 22.043             |

### Embedding Matrices

| Model       | Doctor                                        | Bold                                   | Brave                                  | Dog                                   |
|-------------|-----------------------------------------------|----------------------------------------|----------------------------------------|---------------------------------------|
| RNN         | doctor, troutina, girl, she, queen            | bold, her, she, grandmother, queen     | brave, her, merry, bold, naughty       | girl, cat, dog, she, stepmother       |
| Transformer | she, imaginable, recollecting, knot, skirt    | bold, lapland, compel, shelter, herself| brave, herself, she, compel, her       | she, herself, shelter, her, compel    |
| Bi-LSTM     | doctor, amuse, extraordinarily, active, addressed | she, forgetting, overjoyed, clever, active | addressed, entreated, whispered, angrily, obstacle | active, sob, wept, submissively, lodging |

### Q Values
Q values increased after adding target words, indicating gender bias in all three models.

## Challenges

1. Learning and applying the MLM task
2. Model performance inconsistencies
3. Expanding the dataset and refining preprocessing

## Discussion & Future Work

Fairy tales significantly influence perceptions, especially in children. Our analysis of gender bias in these stories can promote more inclusive storytelling. Future work can:
- Expand the dataset to include contemporary works
- Compare biases across different cultures
- Investigate other biases such as race or socio-economic status

## Final Reflection

Our project successfully trained models and performed bias analysis. We expanded our approach and improved our methodology over time, yielding better results. Key lessons learned include the importance of recognizing bias in training data and the value of both quantitative and qualitative analysis.

## References

- Kurita, K., Vyas, N., Pareek, A., Black, A. W., & Tsvetkov, Y. (2019). Measuring bias in contextualized word representations. Proceedings of the First Workshop on Gender Bias in Natural Language Processing. https://doi.org/10.18653/v1/w19-3823
- Schröder, S., Schulz, A., Kenneweg, P., Feldhans, R., Hinder, F., & Hammer, B. (n.d.). Evaluating Metrics for Bias in Word Embeddings. https://doi.org/10.48550/arXiv.2111.07864

---

This README provides a concise overview of your project, including the methodology, results, challenges, and future work, making it an excellent addition to your portfolio.
