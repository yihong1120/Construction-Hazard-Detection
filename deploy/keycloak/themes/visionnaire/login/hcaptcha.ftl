<#import "template.ftl" as layout>
<@layout.registrationLayout displayMessage=true; section>
    <#if section = "header">
        ${msg("hcaptchaHeading")}
    <#elseif section = "form">
        <form id="kc-hcaptcha-form" action="${url.loginAction}" method="post">
            <div class="h-captcha" data-sitekey="${hcaptchaSiteKey}"></div>
            <div id="kc-form-buttons" class="${properties.kcFormButtonsClass!}">
                <input class="${properties.kcButtonClass!} ${properties.kcButtonPrimaryClass!} ${properties.kcButtonBlockClass!} ${properties.kcButtonLargeClass!}"
                       name="login" id="kc-login" type="submit" value="${msg("doContinue")}"/>
            </div>
        </form>
        <script src="https://js.hcaptcha.com/1/api.js" async defer></script>
    </#if>
</@layout.registrationLayout>
