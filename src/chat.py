import os
import shutil
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from load_data import load_documents      
from vector_database import FaissVectorStore  

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir_profile: str = "faiss_profile", persist_dir_user: str = "faiss_user"):
        print("🛡️ [Init] Initializing RAG Engines...")
        
        self.embedding_model = "all-MiniLM-L6-v2"
        self.user_db_path = persist_dir_user

        # --- BRAIN 1: YOUR PROFILE (Permanent) ---
        self.profile_store = FaissVectorStore(persist_dir_profile, self.embedding_model)
        my_profile_path = "data/MyData.md"
        profile_db_exists = os.path.exists(os.path.join(persist_dir_profile, "faiss.index"))
        
        should_rebuild = False
        
        if profile_db_exists and os.path.exists(my_profile_path):
            file_mtime = os.path.getmtime(my_profile_path)
            db_mtime = os.path.getmtime(persist_dir_profile)
            
            if file_mtime > db_mtime:
                print("🔄 Detect change in MyData.md. Rebuilding database...")
                should_rebuild = True
        
        if not profile_db_exists or should_rebuild:
            if os.path.exists(my_profile_path):
                print(f"📂 Loading profile file: {my_profile_path}")
                if should_rebuild:
                    self.profile_store.clear()
                    self.profile_store = FaissVectorStore(persist_dir_profile, self.embedding_model)

                docs = load_documents(my_profile_path)
                self.profile_store.build_from_documents(docs)
                print("✅ Profile DB updated successfully!")
            else:
                print(f"❌ Error: Profile file not found at {my_profile_path}")
        else:
            print("✅ Loading existing Profile DB (No changes detected).")
            self.profile_store.load()

        # --- BRAIN 2: USER UPLOAD (Temporary) ---
        self.user_store = FaissVectorStore(persist_dir_user, self.embedding_model)
        if os.path.exists(os.path.join(persist_dir_user, "faiss.index")):
            self.user_store.load()

        # --- BRAIN 3: THE LLM ---
        self.llm = self._initialize_robust_llm()

    def _initialize_robust_llm(self, temperature=0.1):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: raise ValueError("❌ GROQ_API_KEY not found.")

        config = {
            "temperature": temperature,
            "max_tokens": 512,
            "api_key": api_key,
            "max_retries": 1,
            "model_kwargs": {"top_p": 0.9}
        }

        # Priority List
        primary_model = "llama-3.3-70b-versatile"
        fallback_models = ["llama-3.1-8b-instant", "mixtral-8x7b-32768"]

        print(f"🤖 [LLM] Primary: {primary_model}")
        
        primary = ChatGroq(model=primary_model, **config)
        fallbacks = [ChatGroq(model=m, **config) for m in fallback_models]
        
        return primary.with_fallbacks(fallbacks)

    def process_user_upload(self, file_path_or_obj):
        print("📂 [Upload] Processing new user file...")
        
        # 1. Wipe old data
        self.user_store.clear()
        
        # 2. Re-initialize
        self.user_store = FaissVectorStore(self.user_db_path, self.embedding_model)
        
        # 3. Load & Build
        original_name = file_path_or_obj.name
        docs = load_documents(file_path_or_obj)
        if docs:
            for doc in docs:
                doc.metadata["source"] = original_name
                if "page" not in doc.metadata:
                    doc.metadata["page"] = 1

            self.user_store.build_from_documents(docs)
            return True, f"Successfully indexed {original_name}"
        return False, "Failed to extract text from file."

    def search_and_answer(self, query: str, top_k: int = 6, mode: str = "profile"):
        """
        mode="profile" -> Searches ONLY Profile DB. Acts as Chetan.
        mode="document" -> Searches ONLY User DB. Acts as Analyst.
        """
        docs = []
        active_persona_prompt = ""
        
        # MODE 1: CHAT WITH CHETAN (Profile DB)
        if mode == "profile":
            if not self.profile_store.index:
                return "My profile database isn't ready. Please check logs."
            
            # Simple, Direct Search
            docs = self.profile_store.query(query, top_k=top_k)
            
            active_persona_prompt = (
                "You are Chetan Kamatagi. Answer in the first person ('I', 'my'). "
                "Use the provided context to describe about you. "
                "If the answer is not in the context, say 'I don't recall mentioning that in my profile'."
            )

        # MODE 2: CHAT WITH DOCUMENT (User DB)
        elif mode == "document":
            if not self.user_store.index:
                return "Please upload a document first so I can analyze it."
            
            # Simple, Direct Search (No summarization hacks needed)
            docs = self.user_store.query(query, top_k=top_k)
            
            active_persona_prompt = (
                "You are a helpful AI Assistant analyzing a document uploaded by the user. "
                "Answer strictly based on the provided document context. "
                "Do not assume the persona of the document's author."
            )

        # GENERATION
        if not docs:
            yield "I couldn't find relevant information in the selected source."
            return

        texts = []
        source_list = []
        for r in docs:
            meta = r.get('metadata', {})
            texts.append(meta.get('text', ''))
            src = meta.get('source', 'Unknown')
            page = meta.get('page', 0) + 1
            source_list.append(f"{os.path.basename(src)} (Pg {page + 1})")
        
        context = "\n\n".join(texts)
        unique_sources = sorted(list(set(source_list)))
        system_prompt = (
            f"{active_persona_prompt}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {query}\n\n"
            "ANSWER:"
        )

        try:
            for chunk in self.llm.stream(system_prompt):
                if chunk.content:
                    yield chunk.content
            yield f"\n\n---\n**📚 References:** {', '.join(unique_refs)}"
        except Exception as e:
            return f"❌ Error: {e}"

if __name__ == "__main__":
    bot = RAGSearch()
    
    # --- TEST 1: PROFILE ---
    print("\nTest 1 (Profile):")
    print(bot.search_and_answer("Who is Chetan?", mode="profile"))

    # --- TEST 2: DOCUMENT ---
    # docu = "YOUR_FILE"
    
    # if os.path.exists(docu):
    #     print(f"\n📂 Uploading: {docu}")
    #     success, msg = bot.process_user_upload(docu)
    #     print(msg)
        
    #     if success:
    #         print("\n--- ❓ Asking Question (Mode: Document) ---")
    #         query = "Summarize the uploaded file content."
            
    #         response = bot.search_and_answer(query, mode="document")
    #         print(f"🤖 Answer:\n{response}")
    # else:
    #     print(f"❌ Test file not found: {docu}")