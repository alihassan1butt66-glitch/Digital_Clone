Digital Clone – Hybrid RAG Chatbot
Student Name: Ali Hassan Butt
Student ID: 23018020005
Semester: 7th
Section: SEC A, BSCS
1. Project Overview
This project is an AI-powered Digital Clone built using Python. It answers questions about my background, skills, projects, and study information using a personal knowledge base and a Hybrid Retrieval-Augmented Generation (RAG) system.
2. Technologies Used
•	Python: Main programming language.
•	BM25: Keyword-based document retrieval.
•	Sentence Transformers (all-MiniLM-L6-v2): Converts document chunks into embeddings.
•	NumPy: Calculates cosine similarity for Dense Search.
•	Hybrid Retrieval (RRF): Combines BM25 and Dense Search results.
•	Groq API: Generates AI-based answers using the openai/gpt-oss-20b model.
•	Streamlit: Provides the chatbot user interface and public hosting.
•	GitHub: Stores the project source code and knowledge-base documents.
3. Hosting and Technology Changes
The original assignment suggested deploying the application through Hugging Face Spaces using Gradio.
During implementation, the available Gradio and Docker hosting options on my Hugging Face account were locked behind paid access. Hugging Face Inference Providers also presented payment and model-access limitations.
I initially developed the chatbot interface using Gradio. To avoid paid hosting, I explored other deployment options. Render requested payment-card details during setup, so I did not proceed with it.
I then selected Streamlit Community Cloud, which allowed me to deploy the application using a public GitHub repository without providing payment-card details.
The interface was converted from Gradio to Streamlit, while the main RAG functionality, including BM25, Dense Search, Hybrid Retrieval, and Groq answer generation, was retained.
Final deployment: Streamlit Community Cloud.
4. Retrieval Comparison
Test Question: What are my programming skills?
•	BM25: Retrieved information based on matching keywords.
•	 

•	Dense: Retrieved semantically related information.

 

•	Hybrid: Combined keyword and semantic results using Reciprocal Rank Fusion.
•	 

Result: Hybrid retrieval provided a useful combination of both search methods.
5. Testing
•	Specific Keyword Test: Asked about granting and denying database permissions in SQL Server. Relevant information was retrieved.
•	 

•	Paraphrase Test: Asked about database-related experience using different wording. The chatbot returned relevant information.
•	 

•	Failure Case: When asked about my favourite programming language, the chatbot initially guessed Python without supporting evidence.
•	 

•	Improvement: Updated the AI prompt to use only facts explicitly available in retrieved documents. If information is missing, it responds: "I don't have that in my notes."
•	 

6. Conclusion
The project successfully demonstrates a Hybrid RAG system combining BM25 and Dense retrieval with AI-based answer generation. The application was tested locally and deployed publicly using Streamlit Community Cloud.
Although the hosting platform was changed due to account access and payment limitations, the main retrieval and answer-generation functionality was maintained.
7. Project Links
Live Application: https://digitalclone-u4564e7uonxwnqw5xbj7ma.streamlit.app/
GitHub Repository: https://github.com/alihassan1butt66-glitch/Digital_Clone

