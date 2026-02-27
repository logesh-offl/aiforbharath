use reqwest::Client;
use serde::{Deserialize, Serialize};

use std::error::Error;
use std::io::Cursor;
use rodio::{Decoder, OutputStream, Sink};
use dotenv::dotenv;
use std::env;

// --- STRUCTURES FOR GEMINI API ---
#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<Content>,
    tools: Option<Vec<Tool>>, // Optional, ignoring for now to fix 404
}

#[derive(Serialize, Deserialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize, Deserialize)]
struct Part {
    text: String,
}

#[derive(Serialize)]
struct Tool {
    google_search: serde_json::Value, // Empty object triggers search
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<Candidate>>,
}

#[derive(Deserialize)]
struct Candidate {
    content: Content,
}

// --- STRUCTURES FOR ELEVENLABS API ---
#[derive(Serialize)]
struct TTSRequest {
    text: String,
    model_id: String,
    voice_settings: VoiceSettings,
}

#[derive(Serialize)]
struct VoiceSettings {
    stability: f32,
    similarity_boost: f32,
}

// --- CORE LOGIC ---

async fn get_healing_advice(client: &Client, user_input: &str, api_key: &str) -> Result<String, Box<dyn Error>> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={}",
        api_key
    );

    // The Prompt Logic
    let system_instruction = "You are an emotional healer. Search for similar stories on Reddit/forums. \
    Ignore negative comments. Summarize ONLY the healing, positive advice found. \
    Start with 'I found a situation just like yours...'";

    let combined_prompt = format!("{}\n\nUser Situation: {}", system_instruction, user_input);

    let request_body = GeminiRequest {
        contents: vec![Content {
            parts: vec![Part { text: combined_prompt }],
        }],
        // This enables Gemini to browse the web for real situations
        // tools: vec![Tool { google_search: json!({}) }], 
        tools: None, 
    };

    let resp = client.post(&url)
        .json(&request_body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let error_text = resp.text().await?;
        println!("API Error: Status: {}, Body: {}", status, error_text);
        return Ok(format!("Error calling Gemini: {}", error_text));
    }

    let resp_json: serde_json::Value = resp.json().await?;
    // println!("Raw Response: {:?}", resp_json); // Debugging

    // Manual deserialization attempt to avoid silent failure
    let gemini_resp: GeminiResponse = serde_json::from_value(resp_json.clone())?;

    // Extract text safely
    if let Some(candidates) = gemini_resp.candidates {
        if let Some(first) = candidates.first() {
            if let Some(part) = first.content.parts.first() {
                return Ok(part.text.clone());
            }
        }
    }
    
    Ok("I'm having trouble connecting to the healing network right now, but I hear you.".to_string())
}

async fn speak_advice(client: &Client, text: &str, api_key: &str, voice_id: &str) -> Result<(), Box<dyn Error>> {
    let url = format!("https://api.elevenlabs.io/v1/text-to-speech/{}", voice_id);

    let payload = TTSRequest {
        text: text.to_string(),
        model_id: "eleven_turbo_v2_5".to_string(), // Turbo is faster for real-time
        voice_settings: VoiceSettings {
            stability: 0.5, // Lower stability = more emotion
            similarity_boost: 0.75,
        },
    };

    let resp = client.post(&url)
        .header("xi-api-key", api_key)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await?;

    if resp.status().is_success() {
        println!("Audio received! Playing stream...");
        let audio_bytes = resp.bytes().await?;
        println!("Received {} bytes of healing audio.", audio_bytes.len());
        
        // Play audio directly from memory
        let (_stream, stream_handle) = OutputStream::try_default()?;
        let sink = Sink::try_new(&stream_handle)?;
        
        let cursor = Cursor::new(audio_bytes);
        let source = Decoder::new(cursor)?;
        
        sink.append(source);
        sink.sleep_until_end();
        
    } else {
        println!("Error from ElevenLabs: {:?}", resp.text().await?);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok(); // Load .env file
    let gemini_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let eleven_key = env::var("ELEVEN_API_KEY").expect("ELEVEN_API_KEY must be set");
    let voice_id = "21m00Tcm4TlvDq8ikWAM"; // Example voice ID (Rachel)

    let client = Client::new();

    // 1. Get User Input from CLI args
    let args: Vec<String> = env::args().collect();
    let user_struggle = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "I feel like I'm failing at my job and my boss hates me. I want to quit.".to_string()
    };
    
    println!("User says: {}", user_struggle);

    // 2. Gemini searches internet for positive similar stories
    println!("Finding similar situations and positive advice...");
    let healing_response = get_healing_advice(&client, &user_struggle, &gemini_key).await?;
    println!("\nGemini Suggests:\n{}", healing_response);

    // 3. ElevenLabs speaks it
    println!("\nSynthesizing voice...");
    speak_advice(&client, &healing_response, &eleven_key, voice_id).await?;

    Ok(())
}