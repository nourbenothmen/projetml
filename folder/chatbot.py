def get_bot_response(user_message):
    message = user_message.lower()
    
    if "cours" in message:
        return "Voici les informations sur les cours : [lien vers les cours]"
    elif "inscription" in message:
        return "Vous pouvez vous inscrire via la page Inscriptions."
    else:
        return "Désolé, je n'ai pas compris votre question."
