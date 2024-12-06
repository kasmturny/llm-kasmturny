package org.example.utils;


import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.stereotype.Component;

@Component
@Aspect
public class logutil {


    @Around(value = "execution(* org.example.dao.StudentDao.*(..))")
    public void around(ProceedingJoinPoint joinPoint){
        //前置通知
        System.out.println("前置通知");
        try {
            Object proceed = joinPoint.proceed();//调用目标业务方法
            //返回通知
            System.out.println("返回通知");
        }catch (Throwable throwable){
            throwable.printStackTrace();
            //异常通知
            System.out.println("异常通知");
        }
        //后置通知
        System.out.println("后置通知");
    }


}
