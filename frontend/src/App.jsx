import { useState, useEffect, useRef } from "react"

function App() {

  // Store selected PDF file
  const [file, setFile] = useState(null)

  // Store user question
  const [question, setQuestion] = useState("")

  // Store chat messages
  const [messages, setMessages] = useState([])

  // Loading state for AI response
  const [loading, setLoading] = useState(false)

  // Upload loading state
  const [uploading, setUploading] = useState(false)

  // Reference for auto-scroll
  const chatEndRef = useRef(null)

  useEffect(() => {

    // Scroll to latest message
    chatEndRef.current?.scrollIntoView({
      behavior: "smooth"
    })

  }, [messages])

  // Upload PDF to backend
  const uploadPDF = async () => {

    // Prevent upload if no file selected
    if (!file) {
      alert("Please select a PDF first")
      return
    }

    // FormData helps send files through HTTP
    const formData = new FormData()

    formData.append("file", file)

    try {

      setUploading(true)

      const response = await fetch(
        "http://127.0.0.1:8001/upload-pdf",
        {
          method: "POST",
          body: formData
        }
      )

      const data = await response.json()

      alert("PDF uploaded successfully 🚀")

      setUploading(false)

      console.log(data)

    } catch (error) {

      console.error(error)

      setUploading(false)

      alert("Upload failed")
    }
  }


  // Ask AI question
  const askQuestion = async () => {

    // Prevent empty questions
    if (!question.trim()) return

    // Add user message to chat
    const userMessage = {
      sender: "user",
      text: question
    }

    setMessages(prev => [...prev, userMessage])

    try {

      setLoading(true)

      const response = await fetch(
        `http://127.0.0.1:8001/ask?question=${question}`,
        {
          method: "POST"
        }
      )

      const data = await response.json()

      // Add AI response
      const aiMessage = {
        sender: "ai",
        text: data.answer || data.error
      }

      setMessages(prev => [...prev, aiMessage])

      // Clear input box
      setQuestion("")

      setLoading(false)

    } catch (error) {

      console.error(error)

      alert("Something went wrong")
      setLoading(false)
    }
  }


  return (

    <div className="min-h-screen bg-zinc-950 text-white">

      {/* Navbar */}
      <div className="border-b border-zinc-800 px-8 py-5">

        <h1 className="text-3xl font-bold">
          NexusAI 🚀
        </h1>

        <p className="text-zinc-400 mt-1">
          AI-powered enterprise knowledge assistant
        </p>

      </div>


      {/* Main Content */}
      <div className="max-w-5xl mx-auto p-6">

        {/* Upload Section */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6 mb-6">

          <h2 className="text-xl font-semibold mb-4">
            Upload PDF
          </h2>

          <input
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
            className="block w-full text-sm text-zinc-300 mb-4"
          />

          <button
            onClick={uploadPDF}
            disabled={uploading}
            className="bg-blue-600 hover:bg-blue-500 px-5 py-3 rounded-xl font-medium cursor-pointer disabled:opacity-50"
          >

            {uploading ? "Uploading..." : "Upload PDF"}

          </button>

        </div>


        {/* Chat Section */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6 h-[500px] flex flex-col">

          <h2 className="text-xl font-semibold mb-4">
            Chat with your document
          </h2>


          {/* Messages */}
          <div className="flex-1 overflow-y-auto space-y-4">

            {messages.map((msg, index) => (

              <div
                key={index}
                className={`
                  p-4 rounded-2xl max-w-[80%] whitespace-pre-wrap shadow-md

                  ${msg.sender === "user"
                    ? "bg-blue-600 ml-auto"
                    : "bg-zinc-800 border border-zinc-700"}
                `}
              >
                {msg.text}
              </div>

            ))}

            {/* AI Thinking Bubble */}
            {loading && (

              <div className="bg-zinc-800 p-4 rounded-2xl max-w-[80%] animate-pulse">

                Thinking...

              </div>

            )}
            <div ref={chatEndRef}></div>

          </div>


          {/* Input Area */}
          <div className="mt-6 flex gap-3">

            <input
              type="text"
              placeholder="Ask something about the PDF..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  askQuestion()
                }
              }}
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-3 outline-none"
            />

            <button
              onClick={askQuestion}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-500 px-6 py-3 rounded-xl font-medium cursor-pointer disabled:opacity-50"
            >
              {loading ? "Thinking..." : "Send"}
            </button>

          </div>

        </div>

      </div>

    </div>
  )
}

export default App