# Digital Clone – Hybrid RAG Chatbot

**Student Name:** Ali Hassan Butt  
**Student ID:** 23018020005  
**Semester:** 7th  
**Section:** SEC A – BSCS  

## 1. Project Overview

This project is a personal AI digital clone built using Retrieval-Augmented Generation (RAG). It answers questions about my background, skills, projects, and learning experience using information stored in my personal knowledge base.

The knowledge base contains six documents:

- `bio.md` – Personal background
- `cv.md` – Education and skills
- `projects.md` – Academic and technical projects
- `opinions.md` – Personal opinions and preferences
- `faq.md` – Frequently asked questions
- `study_notes.md` – Learning and study experiences

These documents were selected to provide different types of personal information and allow the chatbot to retrieve relevant facts.

## 2. Technologies Used

- Python
- Streamlit
- Sentence Transformers (`all-MiniLM-L6-v2`)
- BM25 keyword retrieval
- Dense vector retrieval
- Reciprocal Rank Fusion (RRF)
- Groq API
- GitHub and Streamlit Community Cloud

Documents are divided into chunks of approximately 250 words with 50-word overlap. Source filenames and chunk IDs are retained.

## 3. Retrieval Comparison

The following table compares the results of BM25, Dense, and Hybrid retrieval using five test questions. Retrieved documents are shown with their chunk IDs.

| Test Question | BM25 Results | Dense Results | Hybrid Results | Best Mode |
|---|---|---|---|---|
| 1. What are my programming skills? | faq (0, 2, 1), cv (0) | bio (0), cv (0), opinions (0), faq (0), projects (0) | faq (0), cv (0), opinions (0), bio (0), faq (2) | Hybrid – combined relevant results and gave a concise answer. |
| 2. What did I learn about granting and denying database permissions in SQL Server? | projects (0), faq (0), study_notes (0), cv (0), faq (1) | study_notes (2), faq (2), faq (1), study_notes (0), faq (3) | study_notes (0), faq (2), faq (1), study_notes (2), projects (0) | Dense – returned a detailed explanation of granting, denying, revoking, and database security. |
| 3. Which tool did I use for database work? | faq (0, 1), study_notes (0), cv (0), opinions (0) | study_notes (2), faq (2, 1), study_notes (0) | faq (1), study_notes (0), faq (0, 2), study_notes (2) | BM25 – retrieved the relevant FAQ chunk first and gave the correct answer. |
| 4. What projects have I worked on? | projects (0, 1), faq (2, 0), study_notes (0) | study_notes (0, 1), faq (1, 0), cv (0) | study_notes (0), projects (0), faq (0, 1), study_notes (1) | BM25 – retrieved both project chunks and generated the project answer. |
| 5. What is my name? | faq (0), bio (0), faq (2, 1), projects (1) | faq (2, 0), opinions (1, 0), projects (0) | faq (0, 2), opinions (0), faq (1), projects (1) | BM25 – retrieved the personal information and returned the correct name. |

**Note:** For Q4, Dense and Hybrid retrieved documents but did not generate an answer. This is recorded as a blank response rather than a successful answer.

## 4. Special Retrieval Tests

**Rare Keyword Test**

Question: What did I learn about granting and denying database permissions in SQL Server?

This question tested retrieval of specific terms such as SQL Server, granting, denying, INSERT, and DELETE.

**Paraphrased Test**

Question: Name the relational system I practiced with.

Answer: SQL Server.

This question tested whether the chatbot could understand a question phrased differently from the information stored in the knowledge base.

## 5. Failure Case

Question: What is my favourite programming language and why?

The chatbot initially made an unsupported guess instead of checking whether the information existed in the knowledge base.

**Reason:** The original prompt did not sufficiently prevent the model from making assumptions.

**Improvement made:** I updated the prompt to require answers to be based only on retrieved information. When information is unavailable, the chatbot must respond:

"I don't have that in my notes."

After this change, the chatbot correctly returned the missing-information response.

## 6. Deployment

The original application was developed using Gradio. However, the Gradio and Docker hosting options available to me on Hugging Face required paid access. I therefore converted the interface to Streamlit and deployed it using Streamlit Community Cloud.

The RAG retrieval pipeline was retained, including BM25, Dense, and Hybrid modes.

**Live Application:** : https://digitalclone-u4564e7uonxwnqw5xbj7ma.streamlit.app/

**GitHub Repository:** https://github.com/alihassan1butt66-glitch/Digital_Clone

## 7. Future Improvement

One improvement would be to add more verified personal documents and expand the evaluation questions. This could improve retrieval accuracy and help the chatbot provide more complete answers.

## 8. Conclusion

Through this project, I learned how to build a personal RAG chatbot, prepare a knowledge base, implement BM25 and Dense retrieval, combine results using Reciprocal Rank Fusion, and deploy an interactive application. I also learned the importance of grounded answers and preventing unsupported responses.
