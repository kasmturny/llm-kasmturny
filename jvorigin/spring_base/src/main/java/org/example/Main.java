package org.example;

import org.example.entity.Student;
import org.example.service.StudentService;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

import java.util.List;

/*
 * @description
 * @author： Admin
 * @create： 2024/9/9 16:16
 */
// 按两次 Shift 打开“随处搜索”对话框并输入 `show whitespaces`，
// 然后按 Enter 键。现在，您可以在代码中看到空格字符。
public class Main {
    public static void main(String[] args) {
        //ApplicationContext是应用程序上下文的顶层接口，它有很多种实现，这里我们先介绍第一种
        //因为这里使用的是XML配置文件，所以说我们就使用 ClassPathXmlApplicationContext 这个实现类
        ApplicationContext context = new ClassPathXmlApplicationContext("spring.xml");  //这里写上刚刚的名字
        StudentService studentService = (StudentService) context.getBean("studentService", StudentService.class);
        Student student = (Student) context.getBean("student",Student.class);

        student.setAge(20);
        student.setName("kiana");
        student.setId(2);
        Student student1 = studentService.searchById(1);
        System.out.println(student1);
    }
}