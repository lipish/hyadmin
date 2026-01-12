use axum::{
    extract::Form,
    response::{Html, Redirect},
};

#[derive(serde::Deserialize)]
pub struct LoginForm {
    username: String,
    password: String,
}

pub async fn login_page() -> Html<String> {
    let html = r#"
<!DOCTYPE html>
<html>
<head>
    <title>Heyi Admin - Login</title>
    <style>
        body { font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #f5f5f5; }
        .login-form { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); width: 300px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; }
        input { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .error { color: red; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="login-form">
        <h2>Heyi Admin Login</h2>
        <form method="POST" action="/login">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit">Login</button>
        </form>
    </div>
</body>
</html>
"#;

    Html(html.to_string())
}

pub async fn login(
    Form(form): Form<LoginForm>,
) -> Result<Redirect, axum::http::StatusCode> {
    // Simple authentication (in production, use proper auth)
    if form.username == "admin" && form.password == "admin" {
        // In a real implementation, you'd set a session cookie here
        Ok(Redirect::to("/"))
    } else {
        Err(axum::http::StatusCode::UNAUTHORIZED)
    }
}

pub async fn logout() -> Redirect {
    // Clear session
    Redirect::to("/login")
}
