package org.llm.jvorigin.controller;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
@Controller
public class HelloController {
    @RequestMapping("/")
    @ResponseBody
    public String hello(){
        return "我是兔子兔子兔子";
    }
}

